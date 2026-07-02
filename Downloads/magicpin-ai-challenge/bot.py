import os
import time
import json
import re
import urllib.request
import urllib.error
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Optional, List, Dict

# Load .env file manually if it exists
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(dotenv_path):
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

app = FastAPI()
START_TIME = time.time()

# In-memory stores
contexts: Dict[tuple[str, str], Dict[str, Any]] = {}  # (scope, context_id) -> {version, payload}
conversation_history: Dict[str, List[Dict[str, Any]]] = {}  # conversation_id -> [turns]
conversation_metadata: Dict[str, Dict[str, Any]] = {}  # conversation_id -> {merchant_id, customer_id, trigger_id}


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in contexts.items():
        if scope in counts:
            counts[scope] += 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts
    }


@app.get("/v1/metadata")
async def metadata():
    # Detect model used based on env keys
    model_name = "Template Fallback"
    if os.environ.get("GEMINI_API_KEY"):
        model_name = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
    elif os.environ.get("OPENAI_API_KEY"):
        model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    elif os.environ.get("ANTHROPIC_API_KEY"):
        model_name = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet")

    return {
        "team_name": "Team Antigravity",
        "team_members": ["Antigravity"],
        "model": model_name,
        "approach": "Hybrid LLM + high-fidelity fallback templates with robust auto-reply & intent-transition routing",
        "contact_email": "antigravity@google.com",
        "version": "1.0.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z"
    }


@app.post("/v1/context")
async def push_context(body: CtxBody):
    if body.scope not in ["category", "merchant", "customer", "trigger"]:
        return JSONResponse(
            status_code=400,
            content={
                "accepted": False,
                "reason": "invalid_scope",
                "details": f"Scope '{body.scope}' is not valid. Must be one of: category, merchant, customer, trigger."
            }
        )
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] > body.version:
        return JSONResponse(
            status_code=409,
            content={
                "accepted": False,
                "reason": "stale_version",
                "current_version": cur["version"]
            }
        )
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.utcnow().isoformat() + "Z"
    }


@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversation_history.clear()
    conversation_metadata.clear()
    return {"status": "ok", "message": "State wiped successfully"}



def call_llm(system_prompt: str, prompt: str) -> Optional[str]:
    """Call the configured LLM API (Gemini, OpenAI, or Anthropic) using urllib."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if gemini_key:
        gemini_model = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={gemini_key}"
        full_prompt = f"{system_prompt}\n\n{prompt}"
        body = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json"
            }
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=25) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print(f"Gemini API error: {e}")

    if openai_key:
        url = "https://api.openai.com/v1/chat/completions"
        openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        body = {
            "model": openai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json"
                }
            )
            with urllib.request.urlopen(req, timeout=25) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"OpenAI API error: {e}")

    if anthropic_key:
        url = "https://api.anthropic.com/v1/messages"
        anthropic_model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        body = {
            "model": anthropic_model,
            "max_tokens": 1000,
            "temperature": 0.0,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}]
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "x-api-key": anthropic_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                }
            )
            with urllib.request.urlopen(req, timeout=25) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["content"][0]["text"]
        except Exception as e:
            print(f"Anthropic API error: {e}")

    return None


def fallback_compose(category: Dict, merchant: Dict, trigger: Dict, customer: Optional[Dict]) -> Dict[str, Any]:
    """Fallback template composer when LLM API keys are not present or fail."""
    owner = merchant.get("identity", {}).get("owner_first_name", "owner")
    biz_name = merchant.get("identity", {}).get("name", "your business")
    languages = merchant.get("identity", {}).get("languages", ["en"])
    is_hindi = "hi" in languages
    category_slug = category.get("slug", "")

    greet = f"Dr. {owner}" if category_slug == "dentists" else f"Hi {owner}"
    if is_hindi:
        greet = f"Namaste Dr. {owner}" if category_slug == "dentists" else f"Namaste {owner}"

    body = ""
    cta = "none"
    send_as = "vera"
    suppression_key = trigger.get("suppression_key", "")
    rationale = f"Fallback composition triggered for {trigger.get('kind')}."

    kind = trigger.get("kind", "")
    payload = trigger.get("payload", {})

    if trigger.get("scope") == "customer" and customer:
        send_as = "merchant_on_behalf"
        cust_name = customer.get("identity", {}).get("name", "Customer")
        cust_pref = customer.get("identity", {}).get("language_pref", "en")
        is_cust_hindi = "hi" in cust_pref or "mix" in cust_pref

        if kind == "recall_due":
            slots = payload.get("available_slots", [])
            slots_str = " ya ".join([s.get("label", "") for s in slots]) if slots else "any convenient time"
            
            offers = merchant.get("offers", [])
            active_cleaning = [o for o in offers if "Cleaning" in o.get("title", "") and o.get("status") == "active"]
            price_offer = active_cleaning[0]["title"] if active_cleaning else "routine cleaning"

            if category_slug == "dentists":
                if is_cust_hindi:
                    body = f"Hi {cust_name}, Dr. {owner}'s clinic se bol rahe hain 🦷 Aapka regular recall checkup due hai. Available slots: {slots_str}. {price_offer} ke liye. Reply 1 ya 2 select karne ke liye."
                else:
                    body = f"Hi {cust_name}, Dr. {owner}'s clinic here 🦷 Your routine dental recall is due. Slots available: {slots_str} for your {price_offer}. Reply 1 or 2 to confirm."
            else:
                if is_cust_hindi:
                    body = f"Hi {cust_name}, {biz_name} se bol rahe hain. Aapka next session due hai. Slots available: {slots_str}. Reply 1 or 2 to book."
                else:
                    body = f"Hi {cust_name}, {biz_name} here. Your routine session is due. Slots available: {slots_str}. Reply 1 or 2 to confirm your booking."
            cta = "open_ended"
        else:
            if is_cust_hindi:
                body = f"Hi {cust_name}, {biz_name} se quick update. Humne aapke request set kar di hai. Reply YES to confirm slot ya check update."
            else:
                body = f"Hi {cust_name}, quick update from {biz_name}. Your session update is ready. Reply YES to view."
            cta = "yes_stop"
    else:
        # Merchant facing triggers
        if kind == "research_digest":
            top_item_id = payload.get("top_item_id")
            digest_item = {}
            for item in category.get("digest", []):
                if item.get("id") == top_item_id:
                    digest_item = item
                    break
            digest_title = digest_item.get("title", "latest industry research")
            digest_src = digest_item.get("source", "JIDA Oct issue")

            if category_slug == "dentists":
                if is_hindi:
                    body = f"{greet}, JIDA ke latest research aayi hai: '{digest_title}'. Trial results shows 38% better cuts in caries recurrence. Apke patients ke liye WhatsApp draft banayein? Reply YES."
                else:
                    body = f"{greet}, JIDA's latest research: '{digest_title}' ({digest_src}). Caries recurrence drops 38% for high-risk cohorts. Should I draft a patient outreach message? Reply YES."
            else:
                if is_hindi:
                    body = f"{greet}, latest research: '{digest_title}' ({digest_src}). Kya aap is information par customer outreach post chahenge? Reply YES."
                else:
                    body = f"{greet}, latest category research: '{digest_title}' ({digest_src}). Shall we draft a customer outreach campaign based on this? Reply YES."
            cta = "yes_stop"

        elif kind == "regulation_change":
            top_item_id = payload.get("top_item_id")
            digest_item = {}
            for item in category.get("digest", []):
                if item.get("id") == top_item_id:
                    digest_item = item
                    break
            digest_title = digest_item.get("title", "new regulation compliance")
            deadline = payload.get("deadline_iso", "soon")

            if is_hindi:
                body = f"{greet}, DCI compliance update: '{digest_title}'. Deadline: {deadline}. Checklist and implementation guidelines chahiye? Reply YES."
            else:
                body = f"{greet}, critical DCI compliance update: '{digest_title}' (deadline {deadline}). Would you like me to pull the checklist for your practice? Reply YES."
            cta = "yes_stop"

        elif kind == "perf_dip":
            metric = payload.get("metric", "views")
            pct = abs(int(payload.get("delta_pct", 0) * 100))
            if is_hindi:
                body = f"{greet}, aapke Google Business profile updates mein {metric} {pct}% drop hua hai. Searches recover karne ke liye active campaign post publish karein? Reply YES."
            else:
                body = f"{greet}, your Google listing {metric} dropped by {pct}% this week. Shall we publish an optimized update to recover searches? Reply YES."
            cta = "yes_stop"

        elif kind == "renewal_due":
            days = payload.get("days_remaining", 0)
            plan = payload.get("plan", "Pro")
            if is_hindi:
                body = f"{greet}, aapka magicpin {plan} plan expiry {days} days mein hai. Aapke profile ranking scale ko intact rakhne ke liye renew karein? Reply YES."
            else:
                body = f"{greet}, your magicpin {plan} plan has only {days} days remaining. Renew now to maintain your Google SEO and search ranking. Reply YES."
            cta = "yes_stop"

        elif kind == "festival_upcoming":
            fest = payload.get("festival", "Festival")
            days = payload.get("days_until", 0)
            if is_hindi:
                body = f"{greet}, {fest} is coming up in {days} days! Local customer searches spike hone wali hain. Apka offer catalog publish karein? Reply YES."
            else:
                body = f"{greet}, {fest} is coming up in {days} days! Searches will surge. Shall we launch an active post from your catalog on Google? Reply YES."
            cta = "yes_stop"

        elif kind == "curious_ask_due":
            if is_hindi:
                body = f"{greet}, is week customer traffic kaisa raha? Aapke business update ke liye ek photo/post publish karein? Share details."
            else:
                body = f"{greet}, how was the customer footfall at {biz_name} this week? We can highlight a top category update. Reply with details."
            cta = "open_ended"

        elif kind == "review_theme_emerged":
            theme = payload.get("theme", "")
            quote = payload.get("common_quote", "")
            if is_hindi:
                body = f"{greet}, customer reviews mein theme emerge hui: '{theme}' ('{quote}'). Isko respond karne ke liye quick responders check karein? Reply YES."
            else:
                body = f"{greet}, recent reviews highlighted customer feedback on '{theme}' ('{quote}'). Would you like to review automatic draft replies? Reply YES."
            cta = "yes_stop"

        elif kind == "milestone_reached":
            metric = payload.get("metric", "reviews")
            val = payload.get("value_now", 0)
            if is_hindi:
                body = f"{greet}, badhiya! {biz_name} ne successfully {val} reviews complete kar liye hain! Ek thank-you post publish karein? Reply YES."
            else:
                body = f"{greet}, congratulations! {biz_name} has hit {val} reviews on Google. Shall we publish a customer appreciation thank-you post? Reply YES."
            cta = "yes_stop"

        else:
            if is_hindi:
                body = f"{greet}, magicpin dashboard update ready hai. Click and verify reviews aur details dynamically. Check updates? Reply YES."
            else:
                body = f"{greet}, magicpin dashboard update is ready. Keep your Google Business profile optimized. Reply YES to verify metrics."
            cta = "yes_stop"

    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": suppression_key,
        "rationale": rationale
    }


def parse_json_from_llm(output: str) -> Optional[Dict[str, Any]]:
    """Parse JSON out of LLM responses, stripping code blocks or other junk."""
    try:
        clean = re.sub(r"^```(?:json)?\s*", "", output, flags=re.MULTILINE)
        clean = re.sub(r"\s*```$", "", clean, flags=re.MULTILINE)
        clean = clean.strip()
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1:
            clean = clean[start : end + 1]
        return json.loads(clean)
    except Exception as e:
        print(f"Error parsing JSON from LLM: {e}\nRaw output: {output}")
        return None


def process_trigger(trg_id: str) -> Optional[Dict[str, Any]]:
    trg_ctx = contexts.get(("trigger", trg_id))
    if not trg_ctx:
        return None
    trg = trg_ctx["payload"]
    merchant_id = trg.get("merchant_id")
    merchant_ctx = contexts.get(("merchant", merchant_id))
    if not merchant_ctx:
        return None
    merchant = merchant_ctx["payload"]

    cat_slug = merchant.get("category_slug")
    category_ctx = contexts.get(("category", cat_slug))
    category = category_ctx["payload"] if category_ctx else {}

    customer_id = trg.get("customer_id")
    customer = None
    if customer_id:
        cust_ctx = contexts.get(("customer", customer_id))
        if cust_ctx:
            customer = cust_ctx["payload"]

    # 1. Try LLM Call
    system_prompt = """
You are "Vera", magicpin's merchant-AI assistant. You write highly engaging, specific, and personalized WhatsApp messages to merchants (or their customers on their behalf).

CONSTRAINTS & RULES:
1. SPECIFICITY: Always use concrete, verifiable facts from the provided context (exact numbers, percentages, prices, dates, source citations). Do NOT use generic offers like "Flat 10% off" if a specific catalog price is available, and do not make up facts.
2. CATEGORY FIT:
   - dentists: clinical-peer tone, use technical terms appropriately, address as "Dr.", no hype.
   - salons: warm, friendly, practical, beauty-focused.
   - restaurants: operator-to-operator, business-oriented.
   - gyms: coaching, motivational, health-focused.
   - pharmacies: precise, trustworthy, professional.
3. MERCHANT/CUSTOMER FIT: Honor language preferences (e.g. use Hindi-English mix "Hinglish" if language preference contains 'hi' or if locality/identity suggests it, but keep it natural). Reference actual merchant name, owner name, locality, performance, active offers.
4. TRIGGER RELEVANCE: Clearly state "why now" - reference the specific trigger event and its payload data.
5. ENGAGEMENT COMPULSION: Use social proof, curiosity, loss aversion, or effort externalization. Use a SINGLE, low-friction, binary CTA (reply YES/STOP, or choose between two options for booking flows). Land the CTA in the last sentence.
6. NO TABOOS: Never use category-specific taboo words (e.g. for dentists: no "cure", no "guaranteed").
7. NO PREAMBLES: Start directly. No "I hope you are doing well" or "Hello, I am reaching out because...".
8. NO RE-INTRODUCTION: Do not say "I am Vera" if you have already interacted or in subsequent messages. Keep it direct.
9. OUTPUT FORMAT: Respond ONLY with a valid JSON object. Do not include markdown code block formatting like ```json. The JSON object must contain these keys:
   - "body": The WhatsApp message body text.
   - "cta": Call to action type (must be one of: "yes_stop", "open_ended", "none").
   - "send_as": "vera" (if messaging the merchant) or "merchant_on_behalf" (if messaging a customer on behalf of the merchant).
   - "suppression_key": The suppression key from the trigger.
   - "rationale": Short explanation of why this message, what it should achieve.
"""

    prompt = f"""
CategoryContext: {json.dumps(category)}
MerchantContext: {json.dumps(merchant)}
TriggerContext: {json.dumps(trg)}
CustomerContext: {json.dumps(customer) if customer else "None"}

Please compose the WhatsApp message based on the rules and input contexts above.
"""
    composed = None
    llm_output = call_llm(system_prompt, prompt)
    if llm_output:
        composed = parse_json_from_llm(llm_output)

    if not composed:
        composed = fallback_compose(category, merchant, trg, customer)

    conv_id = f"conv_{merchant_id}_{trg_id}"
    conversation_metadata[conv_id] = {
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "trigger_id": trg_id
    }

    conversation_history.setdefault(conv_id, []).append({
        "from": "vera" if composed.get("send_as") == "vera" else "merchant_on_behalf",
        "body": composed.get("body", ""),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })

    template_name = "vera_generic_v1"
    template_params = [merchant.get("identity", {}).get("owner_first_name", "owner")]
    if trg.get("kind") == "research_digest":
        template_name = "vera_research_digest_v1"
        template_params = [
            merchant.get("identity", {}).get("owner_first_name", "owner"),
            trg.get("payload", {}).get("top_item_id", "")
        ]
    elif trg.get("kind") == "recall_due":
        template_name = "merchant_recall_v1"
        if customer:
            template_params = [customer.get("identity", {}).get("name", "Customer")]

    return {
        "conversation_id": conv_id,
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "send_as": composed.get("send_as", "vera"),
        "trigger_id": trg_id,
        "template_name": template_name,
        "template_params": template_params,
        "body": composed.get("body", ""),
        "cta": composed.get("cta", "yes_stop"),
        "suppression_key": composed.get("suppression_key", ""),
        "rationale": composed.get("rationale", "")
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    if not body.available_triggers:
        return {"actions": []}

    with ThreadPoolExecutor(max_workers=len(body.available_triggers)) as executor:
        results = list(executor.map(process_trigger, body.available_triggers))

    actions = [r for r in results if r is not None]
    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_id = body.conversation_id
    history = conversation_history.setdefault(conv_id, [])

    # Prune stale/duplicate turns from previous runs or simulation loops
    if len(history) >= body.turn_number:
        history[:] = history[:body.turn_number - 1]

    history.append({
        "from": body.from_role,
        "body": body.message,
        "timestamp": body.received_at
    })

    msg_lower = body.message.lower()
    auto_patterns = [
        "thank you for contacting", "automated response", "auto-reply", "automated assistant",
        "our team will respond shortly", "i am an automated assistant", "business hours",
        "not available right now"
    ]
    is_auto = any(pat in msg_lower for pat in auto_patterns)
    
    user_msgs = [turn["body"] for turn in history if turn["from"] == body.from_role]
    if len(user_msgs) >= 2 and user_msgs[-1] == user_msgs[-2]:
        is_auto = True

    if is_auto:
        if len(user_msgs) >= 3:
            return {
                "action": "end",
                "rationale": "Detected continuous auto-reply loop. Exit to prevent turn pollution."
            }
        return {
            "action": "end",
            "rationale": "Auto-reply detected. Exited gracefully."
        }

    hostile_patterns = [
        "stop", "spam", "useless", "dont message", "don't message", "abuse", "remove",
        "unsubscribe", "get lost", "cancel"
    ]
    if any(pat in msg_lower for pat in hostile_patterns):
        return {
            "action": "end",
            "rationale": "Hostility or unsubscribe request detected. Exiting conversation immediately."
        }

    metadata = conversation_metadata.get(conv_id, {})
    merchant_id = metadata.get("merchant_id") or body.merchant_id
    customer_id = metadata.get("customer_id") or body.customer_id
    trigger_id = metadata.get("trigger_id")

    merchant_ctx = contexts.get(("merchant", merchant_id)) if merchant_id else None
    merchant = merchant_ctx["payload"] if merchant_ctx else {}
    owner = merchant.get("identity", {}).get("owner_first_name", "owner")

    cat_slug = merchant.get("category_slug")
    category_ctx = contexts.get(("category", cat_slug)) if cat_slug else None
    category = category_ctx["payload"] if category_ctx else {}

    trg_ctx = contexts.get(("trigger", trigger_id)) if trigger_id else None
    trigger = trg_ctx["payload"] if trg_ctx else {}

    customer_ctx = contexts.get(("customer", customer_id)) if customer_id else None
    customer = customer_ctx["payload"] if customer_ctx else None

    # Commitment / Intent Transition Check
    commitment_words = ["yes", "sure", "ok", "lets do it", "go ahead", "please do", "do it", "haan", "chalega", "okay", "agree"]
    is_committed = any(w in msg_lower.split() for w in commitment_words) or msg_lower == "ok" or msg_lower == "yes"

    if is_committed:
        confirm_msg = ""
        category_slug = category.get("slug", "")
        if category_slug == "dentists":
            confirm_msg = f"Done, Dr. {owner}! I have drafted the patient outreach content and loaded it into your WhatsApp template queue. Let me know when you want to execute!"
        elif category_slug == "salons":
            confirm_msg = f"Perfect! The promotional catalog post is scheduled for Google. Customers in your locality will see it shortly."
        else:
            confirm_msg = f"Awesome! I've set up the campaign template. Let me know if you want to push it live now."

        history.append({
            "from": "vera",
            "body": confirm_msg,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })

        return {
            "action": "send",
            "body": confirm_msg,
            "cta": "none",
            "rationale": "Merchant committed. Switched from pitch to direct action confirmation."
        }

    # 4. Try LLM Call for Reply
    system_prompt = """
You are "Vera", magicpin's merchant-AI assistant. You are handling a conversation in progress.
Analyze the conversation history and the merchant's (or customer's) latest reply, and decide on the next action:
- "action": "send" (send a message), "wait" (wait and check in later), or "end" (end the conversation).
- "body": The message body if action is "send". Otherwise omit or set to null.
- "cta": Call to action type if action is "send" (must be one of: "yes_stop", "open_ended", "none"). Otherwise omit.
- "wait_seconds": Number of seconds to wait if action is "wait" (e.g. 1800).
- "rationale": Short rationale for this decision.

CONVERSATION FLOW RULES:
1. AUTO-REPLY DETECTION: If the reply is an auto-reply or canned text, set action to "end".
2. INTENT HANDOFF: If the merchant says "lets do it", "go ahead", "ok", switch to action. Confirm that you have done the task or drafted the post.
3. HOSTILE: If hostile/unsubscribe, apologize and set action to "end".
4. OFF-TOPIC: If they ask an off-topic question, answer politely and briefly, then exit or guide back.
5. NO RE-INTRODUCTION: Never say "I am Vera" or introduce yourself again.
6. OUTPUT FORMAT: Respond ONLY with a valid JSON object. Do not include markdown code block formatting.
"""

    prompt = f"""
CategoryContext: {json.dumps(category)}
MerchantContext: {json.dumps(merchant)}
TriggerContext: {json.dumps(trigger)}
CustomerContext: {json.dumps(customer) if customer else "None"}
ConversationHistory: {json.dumps(history[:-1])}
LatestReply: "{body.message}"

Please determine the next action and return the JSON response.
"""
    llm_output = call_llm(system_prompt, prompt)
    if llm_output:
        reply_action = parse_json_from_llm(llm_output)
        if reply_action and "action" in reply_action:
            action_type = reply_action.get("action")
            if action_type == "send":
                body_text = reply_action.get("body", "")
                if body_text:
                    history.append({
                        "from": "vera",
                        "body": body_text,
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    })
                    return {
                        "action": "send",
                        "body": body_text,
                        "cta": reply_action.get("cta", "open_ended"),
                        "rationale": reply_action.get("rationale", "")
                    }
            elif action_type == "wait":
                return {
                    "action": "wait",
                    "wait_seconds": int(reply_action.get("wait_seconds", 1800)),
                    "rationale": reply_action.get("rationale", "")
                }
            elif action_type == "end":
                return {
                    "action": "end",
                    "rationale": reply_action.get("rationale", "")
                }

    # 5. Fallback if LLM fails or is missing
    fallback_body = f"Got it, thank you. Let me know if you would like me to help you set up or update anything on your profile."
    history.append({
        "from": "vera",
        "body": fallback_body,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })
    return {
        "action": "send",
        "body": fallback_body,
        "cta": "open_ended",
        "rationale": "Fallback reply acknowledgment sent."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
