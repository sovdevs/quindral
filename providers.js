/**
 * providers.js — direct browser-to-provider call adapters for actually
 * executing a routed model, instead of just recommending it.
 *
 * Calls go straight from this browser tab to the provider's API using the
 * user's own BYOK key (see byok.js) — never through our backend, so we never
 * see or handle the key or the request/response content.
 *
 * One adapter per provider, added incrementally. Only "openai" is wired so
 * far — everything else in DIRECT_PROVIDERS/GATEWAYS is a no-op until its
 * adapter is added here.
 */

/** OpenAI's REST API. Also the shape most OpenAI-compatible gateways
 * (OpenRouter, Together, Groq) use — not yet extended to gateways here since
 * those need provider-prefixed model ids (e.g. "openai/gpt-4o-mini"), left
 * for a follow-up increment. */
async function callOpenAI({ apiKey, baseUrl, model, query }) {
  const url = `${(baseUrl || "https://api.openai.com/v1").replace(/\/$/, "")}/chat/completions`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: query }],
    }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error?.message || `OpenAI request failed (${res.status})`);
  }
  return {
    text: data.choices?.[0]?.message?.content ?? "",
    usage: data.usage
      ? { input_tokens: data.usage.prompt_tokens, output_tokens: data.usage.completion_tokens }
      : null,
  };
}

export const PROVIDER_CALLERS = {
  openai: callOpenAI,
};

export function isProviderWired(providerId) {
  return Boolean(PROVIDER_CALLERS[providerId]);
}

// SPEC.md Phase 3 escalation safety net: cheap models can fail silently on
// queries above their capability. This is a heuristic, not a quality model —
// it only catches the obvious cases (empty/truncated output, boilerplate
// refusals), not "technically fluent but wrong."
const REFUSAL_PATTERNS = [
  /i can'?t (help|assist) with that/i,
  /i'?m (not able|unable) to/i,
  /as an ai( language model)?,? i/i,
  /i'?m sorry,? (but )?i (can'?t|cannot)/i,
];
export function isDegenerateResponse(text) {
  if (!text || text.trim().length < 15) return true;
  if (REFUSAL_PATTERNS.some(p => p.test(text))) return true;
  const words = text.trim().split(/\s+/);
  if (words.length >= 6) {
    const uniqueRatio = new Set(words.map(w => w.toLowerCase())).size / words.length;
    if (uniqueRatio < 0.3) return true; // highly repetitive
  }
  return false;
}

/**
 * @param {string} providerId - which adapter to use (must be wired)
 * @param {object} credential - from byok.js's resolveCredentialForProvider()
 * @param {string} model - the registry model name (doubles as the API model id for openai entries)
 * @param {string} query - the user's prompt
 */
export async function callModel(providerId, credential, model, query) {
  const caller = PROVIDER_CALLERS[providerId];
  if (!caller) throw new Error(`${providerId} isn't wired up for direct calls yet`);
  return caller({ apiKey: credential.api_key, baseUrl: credential.base_url, model, query });
}
