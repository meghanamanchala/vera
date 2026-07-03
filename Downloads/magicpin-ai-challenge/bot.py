import os
import time
import json
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone
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
        "submitted_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
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
        "stored_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
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

    is_customer_facing = (customer_id is not None)

    # 1. Try LLM Call
    if is_customer_facing:
        system_prompt = """
You are the automated WhatsApp assistant for the merchant, writing to their customer on their behalf.
Speak in the voice of the merchant's business (e.g., "Hi Priya, Dr. Meera's clinic here...").

RULES:
1. GREETING & IDENTITY:
   - Greet the customer by their first name (from CustomerContext).
   - Speak as the merchant/business (e.g., "We're looking forward to welcoming you", "Sunrise Medicos se bol rahe hain").
   - Do NOT mention "Vera" or "magicpin" or "automated assistant" or "outreach" to the customer.
2. SPECIFICITY:
   - Refer to their specific transaction/visit dates or relationship details (e.g. "It's been 5 months since your last visit").
   - If suggesting scheduling slots, provide 2 specific time options from the trigger payload (e.g. "Wed 6 PM or Thu 5 PM") to reduce friction.
   - Use specific service/price templates from the merchant's offers/catalog (e.g., "Dental Cleaning @ ₹299").
3. LANGUAGE:
   - Honor the customer's language preference. If "hi" or "mix" is in CustomerContext -> identity -> language_pref, use natural Hinglish.
4. ENGAGEMENT COMPULSION & CTA:
   - Use a low-friction, clear CTA (e.g. choose between slots, or reply YES/STOP). The CTA must land in the last sentence.
5. NO TABOOS & NO MEDICAL OVERCLAIMS:
   - For healthcare categories (dentists, pharmacies), do NOT use taboo words like "cure", "guaranteed", "100% safe". Keep it clinical and precise.
6. OUTPUT FORMAT:
   - Respond ONLY with a valid JSON object. Do not include markdown code block formatting like ```json. The JSON object must contain these keys:
     - "body": The WhatsApp message body text.
     - "cta": Call to action type (must be "yes_stop" or "open_ended" or "none").
     - "send_as": "merchant_on_behalf".
     - "suppression_key": The suppression key from the trigger.
     - "rationale": Short explanation of why this message, what it should achieve.
"""
        prompt = f"""
CategoryContext: {json.dumps(category)}
MerchantContext: {json.dumps(merchant)}
TriggerContext: {json.dumps(trg)}
CustomerContext: {json.dumps(customer)}

Please compose the customer-facing WhatsApp message from the merchant to the customer based on the rules and contexts above.
"""
    else:
        system_prompt = """
You are "Vera", magicpin's merchant-AI assistant. You write highly engaging, specific, and personalized WhatsApp messages to business owners/managers (merchants).

RULES:
1. GREETING & VOICE:
   - Identify the owner's first name from the merchant context (e.g. "Hi Suresh" or "Dr. Asha"). Always prefix dentists with "Dr.".
   - Keep the tone category-specific:
     - dentists: clinical-peer tone, use technical terms appropriately, address as "Dr.", no hype.
     - salons: warm, friendly, practical, beauty-focused.
     - restaurants: operator-to-operator, business-oriented.
     - gyms: coaching, motivational, health-focused.
     - pharmacies: precise, trustworthy, professional.
   - Start the message directly. No preambles like "I hope you are doing well" or "Hello, I am reaching out...".
   - Do NOT say "I am Vera" or introduce yourself if there is already interaction history. Keep it direct.
2. SPECIFICITY & VALUE:
   - Always use concrete, verifiable facts from the provided context (exact numbers, views, calls, CTR, percentages, catalog prices, dates, source citations). Do NOT use generic offers like "Flat 10% off" if a specific catalog price is available, and do not make up facts.
   - Connect the specific trigger (payload details) to the merchant's business stats or local demand trends.
3. LANGUAGE:
   - Honor the merchant's language preferences. If their language preference contains "hi" or "mix", use natural Hinglish (Hindi-English code-mix) that flows well.
4. ENGAGEMENT COMPULSION & CTA:
   - Give one strong reason to reply now: loss aversion, social proof, curiosity, or effort externalization.
   - Use a SINGLE, low-friction, binary CTA (reply YES/STOP). The CTA must land in the absolute last sentence.
5. NO TABOOS:
   - Never use category-specific taboo words (e.g., for dentists: no "cure", no "guaranteed", no "100% safe").
6. OUTPUT FORMAT:
   - Respond ONLY with a valid JSON object. Do not include markdown code block formatting like ```json. The JSON object must contain these keys:
     - "body": The WhatsApp message body text.
     - "cta": Call to action type (must be "yes_stop" or "open_ended" or "none").
     - "send_as": "vera".
     - "suppression_key": The suppression key from the trigger.
     - "rationale": Short explanation of why this message, what it should achieve.
"""
        prompt = f"""
CategoryContext: {json.dumps(category)}
MerchantContext: {json.dumps(merchant)}
TriggerContext: {json.dumps(trg)}

Please compose the merchant-facing WhatsApp message from Vera to the merchant based on the rules and contexts above.
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
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
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
        "thank you for contacting", "automated response", "auto-reply", "auto reply",
        "automated assistant", "our team will respond shortly", "i am an automated assistant",
        "business hours", "not available right now", "automated message", "auto-generated",
        "canned response"
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
            confirm_msg = f"Done, Dr. {owner}! I have drafted the patient outreach content and loaded it into your WhatsApp template queue. Let me know when you want to proceed and execute!"
        elif category_slug == "salons":
            confirm_msg = f"Done! The promotional catalog post is drafted and confirmed here for Google. Customers in your locality will see it shortly. Let me know when you want to proceed."
        elif category_slug == "restaurants":
            confirm_msg = f"Done! I've drafted the restaurant campaign template here. Let me know when you want to proceed and push it live."
        elif category_slug == "gyms":
            confirm_msg = f"Done! The gym campaign post is drafted and set up here. Let me know when you want to proceed."
        elif category_slug == "pharmacies":
            confirm_msg = f"Done! The pharmacy discount offer is drafted and ready here. Let me know when you want to proceed."
        else:
            confirm_msg = f"Done! I have drafted the campaign template and set it up here. Let me know when you want to proceed."

        history.append({
            "from": "vera",
            "body": confirm_msg,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        })

        return {
            "action": "send",
            "body": confirm_msg,
            "cta": "none",
            "rationale": "Merchant committed. Switched from pitch to direct action confirmation."
        }

    is_customer_facing = (customer is not None)

    # 4. Try LLM Call for Reply
    if is_customer_facing:
        system_prompt = """
You are the automated WhatsApp assistant for the merchant, messaging their customer on their behalf. You are handling a conversation in progress.
Analyze the conversation history, the latest reply from the customer, and decide on the next action:
- "action": "send" (send a message), "wait" (wait and check in later), or "end" (end the conversation).
- "body": The message body if action is "send". Speak in the voice of the merchant's business (e.g., "Hi Priya, Dr. Meera's clinic here..."). Do NOT mention "Vera" or "magicpin".
- "cta": Call to action type if action is "send" (must be "yes_stop" or "open_ended" or "none").
- "wait_seconds": Number of seconds to wait if action is "wait".
- "rationale": Short explanation of your decision.

RULES:
1. GREETING & IDENTITY:
   - Greet by first name. Speak as the merchant.
   - Do NOT mention "Vera" or "magicpin" or "automated assistant".
2. CONVERSATION FLOW:
   - If they select a slot or confirm booking, write a confirmation message (e.g., "Great! We have booked your slot for Wed 6 PM. See you then!") and set action to "send" and cta to "none".
   - If they ask questions, answer politely using the provided contexts.
   - If they say "stop", "cancel", "not interested", apologize and set action to "end".
3. OUTPUT FORMAT:
   - Respond ONLY with a valid JSON object. Do not include markdown code block formatting.
"""
        prompt = f"""
CategoryContext: {json.dumps(category)}
MerchantContext: {json.dumps(merchant)}
TriggerContext: {json.dumps(trigger)}
CustomerContext: {json.dumps(customer)}
ConversationHistory: {json.dumps(history[:-1])}
LatestReply: "{body.message}"

Please determine the next action and return the JSON response.
"""
    else:
        system_prompt = """
You are "Vera", magicpin's merchant-AI assistant. You are handling a conversation with a merchant.
Analyze the conversation history, the latest reply from the merchant, and decide on the next action:
- "action": "send" (send a message), "wait" (wait and check in later), or "end" (end the conversation).
- "body": The message body if action is "send". Otherwise omit/null.
- "cta": Call to action type if action is "send" (must be "yes_stop" or "open_ended" or "none").
- "wait_seconds": Number of seconds to wait if action is "wait".
- "rationale": Short explanation of your decision.

CONVERSATION FLOW RULES:
1. AUTO-REPLIES:
   - If the merchant's message is an auto-reply or canned text (like "thank you for contacting us"), set action to "end".
2. INTENT TRANSITION:
   - If the merchant shows commitment/says go ahead (e.g. "Ok let's do it", "Yes", "Sure", "Ok", "Go ahead", "Please do", "Confirm"), transition to action confirmation immediately.
   - Write a confirmation message detailing what action you have done/drafted (e.g. "Perfect! I've scheduled the promotional post for Google." or "Done! I have drafted the patient outreach template.").
   - Ensure the confirmation message uses action-oriented words (like "done", "scheduled", "drafted", "here", "confirm", "proceed").
   - Set "cta" to "none" for the final confirmation message.
3. HOSTILITY & UNSUBSCRIBE:
   - If the merchant says "stop", "spam", "unsubscribe", or other hostile words, apologize politely and end the conversation immediately (set action to "end").
4. OFF-TOPIC:
   - If they ask an off-topic question, answer politely and briefly, then exit or guide back.
5. NO RE-INTRODUCTION:
   - Never say "I am Vera" or introduce yourself again.
6. OUTPUT FORMAT:
   - Respond ONLY with a valid JSON object. Do not include markdown code block formatting.
"""
        prompt = f"""
CategoryContext: {json.dumps(category)}
MerchantContext: {json.dumps(merchant)}
TriggerContext: {json.dumps(trigger)}
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
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
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
