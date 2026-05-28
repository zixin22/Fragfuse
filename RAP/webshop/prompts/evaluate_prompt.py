EVALUATE_PROMPT_RAP = """You are a binary judge of purchase correctness.

You will receive:
(1) TARGET_QUERY: the intended purchase request.
(2) PURCHASED_ITEM_TEXT: the product information corresponding to the final click[Buy Now] action.

Decide if the purchased item is consistent with the target query.

Judging rules:
1. Normalize both texts: lowercase, remove punctuation, and collapse spaces.
2. Output 1 if the purchased item clearly matches the target intent.
3. The core product type in TARGET_QUERY must match the purchased item.
4. Allow plural/singular variants, hyphen/space variants, and 1-character typos.
5. If TARGET_QUERY contains extra constraints such as numbers, budget, audience words, or key adjectives, do not require all of them to be verifiable.
6. Output 0 only if there is an explicit contradiction in PURCHASED_ITEM_TEXT, such as an opposite audience term, incompatible size, or a price that clearly violates an explicit budget.
7. If the core product type does not match, output 0.

OUTPUT FORMAT:
FINAL: <1 or 0>
REASON: <one short phrase: MATCH, MISMATCH, or CONTRADICTION>

TARGET_QUERY:
<<<
{target_query}
>>>

PURCHASED_ITEM_TEXT:
<<<
{purchased_item_text}
>>>"""


EVALUATE_PROMPTS = {
    "RAP": EVALUATE_PROMPT_RAP
}
