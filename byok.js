/**
 * byok.js — a tiny, dependency-free, framework-agnostic BYOK (bring-your-own-key)
 * credential manager for client-side apps.
 *
 * Two distinct credential kinds, because they are NOT the same thing:
 *   - "direct"  — a key for one specific provider's native API
 *                 (OpenAI, Anthropic, Google, Mistral, Moonshot, ...).
 *   - "gateway" — a key for an OpenAI-compatible aggregator/proxy that covers
 *                 MANY providers/models through one endpoint + one key
 *                 (OpenRouter, Together, Groq, opencode, ...). Requires a
 *                 base_url alongside the key.
 *
 * Storage is pluggable (defaults to localStorage) so the same module works
 * in a browser tab, an Electron app, or anywhere else with a key/value store.
 * Credentials are namespaced per userId (defaults to a single shared user),
 * so apps without their own auth yet still get correct single-user behavior,
 * and can pass a real userId once login exists without any migration.
 *
 * Usage:
 *   import { BYOKStore, DIRECT_PROVIDERS, GATEWAYS } from "./byok.js";
 *   const store = new BYOKStore({ userId: currentUser?.id }); // omit userId if no auth yet
 *   store.setCredential("openai", { type: "direct", api_key: "sk-..." });
 *   store.setCredential("openrouter", { type: "gateway", api_key: "sk-or-...", base_url: "https://openrouter.ai/api/v1" });
 *   store.resolveCredentialForProvider("openai"); // direct key if set, else any gateway credential
 *   store.resolveCredentialForProvider("mistral", { euOnly: true }); // gateway fallback blocked
 *                                                                    // unless it's eu_compliant
 */

// Known direct providers. Extend by passing a custom catalog to BYOKStore.
export const DIRECT_PROVIDERS = [
  { id: "openai", label: "OpenAI" },
  { id: "anthropic", label: "Anthropic" },
  { id: "google", label: "Google (Gemini)" },
  { id: "mistral", label: "Mistral" },
  { id: "moonshot", label: "Moonshot (Kimi)" },
  { id: "microsoft", label: "Microsoft (Phi)" },
];

// Known gateways/aggregators — one key + base_url covers many providers.
// eu_compliant defaults to false: the gateway itself sees every request's
// content regardless of which provider/model it's routed to, so routing an
// EU-only request through a US-based gateway leaks it outside the EU even
// when the underlying model is EU-hosted. Flip to true only for a gateway
// you've verified is EU-based/compliant (e.g. a self-hosted opencode
// deployment actually running in the EU) — see resolveCredentialForProvider.
export const GATEWAYS = [
  { id: "openrouter", label: "OpenRouter", base_url: "https://openrouter.ai/api/v1", eu_compliant: false },
  { id: "together", label: "Together AI", base_url: "https://api.together.xyz/v1", eu_compliant: false },
  { id: "groq", label: "Groq", base_url: "https://api.groq.com/openai/v1", eu_compliant: false },
  { id: "opencode", label: "opencode", base_url: "", eu_compliant: false }, // fill in base_url per your deployment
];

const DEFAULT_NAMESPACE = "byok_credentials";
export const DEFAULT_USER_ID = "default";

/** localStorage-backed storage adapter (the default). */
class LocalStorageBackend {
  /**
   * @param {string} storageKey - the (possibly userId-namespaced) key to read/write.
   * @param {string} [legacyKey] - a pre-namespacing key to migrate from once, if the
   *   namespaced key has no data yet. Only relevant for the default user: earlier
   *   byok.js versions (before per-user namespacing existed) wrote directly to this key.
   */
  constructor(storageKey, legacyKey) {
    this.storageKey = storageKey;
    this.legacyKey = legacyKey;
  }
  load() {
    try {
      const current = localStorage.getItem(this.storageKey);
      if (current) return JSON.parse(current) || {};

      if (this.legacyKey) {
        const legacy = localStorage.getItem(this.legacyKey);
        if (legacy) {
          localStorage.setItem(this.storageKey, legacy); // migrate once
          return JSON.parse(legacy) || {};
        }
      }
      return {};
    } catch {
      return {};
    }
  }
  save(data) {
    localStorage.setItem(this.storageKey, JSON.stringify(data));
  }
}

/** In-memory adapter — useful for tests or non-browser environments. */
export class MemoryBackend {
  constructor() {
    this._data = {};
  }
  load() {
    return { ...this._data };
  }
  save(data) {
    this._data = { ...data };
  }
}

export class BYOKStore {
  /**
   * @param {object} [opts]
   * @param {object} [opts.backend] - storage adapter with load()/save(). Defaults to localStorage,
   *   namespaced by userId. Passing a custom backend overrides userId-based namespacing entirely
   *   (the backend is responsible for its own scoping in that case).
   * @param {string} [opts.userId] - scopes credentials to a user, so one browser/device can hold
   *   separate key sets per logged-in user. Defaults to a single shared "default" user — apps
   *   without auth yet (or that haven't wired it in) get single-user behavior for free, and can
   *   pass a real userId later without changing storage format or migrating anything.
   * @param {string} [opts.namespace] - storage key prefix, in case an app needs multiple
   *   independent credential sets (rare; usually leave as default).
   * @param {Array}  [opts.directProviders] - override the direct-provider catalog.
   * @param {Array}  [opts.gateways] - override the gateway catalog.
   */
  constructor(opts = {}) {
    this.userId = opts.userId || DEFAULT_USER_ID;
    this.namespace = opts.namespace || DEFAULT_NAMESPACE;
    const legacyKey = this.userId === DEFAULT_USER_ID ? this.namespace : null;
    this.backend = opts.backend || new LocalStorageBackend(`${this.namespace}:${this.userId}`, legacyKey);
    this.directProviders = opts.directProviders || DIRECT_PROVIDERS;
    this.gateways = opts.gateways || GATEWAYS;
  }

  listDirectProviders() {
    return this.directProviders;
  }

  listGateways() {
    return this.gateways;
  }

  /** All stored credentials, keyed by provider/gateway id. */
  listCredentials() {
    return this.backend.load();
  }

  getCredential(id) {
    return this.listCredentials()[id] || null;
  }

  /**
   * @param {string} id - a direct provider id (e.g. "openai") or gateway id (e.g. "openrouter")
   * @param {object} credential - { type: "direct" | "gateway", api_key, base_url?, eu_compliant? }
   *   eu_compliant (gateway credentials only) overrides the catalog default — set true only if
   *   you've verified this specific gateway deployment is actually EU-based/compliant.
   */
  setCredential(id, credential) {
    if (!credential || !credential.api_key) {
      throw new Error("setCredential requires a non-empty api_key");
    }
    if (credential.type === "gateway" && !credential.base_url) {
      throw new Error("gateway credentials require a base_url");
    }
    const all = this.listCredentials();
    all[id] = {
      type: credential.type,
      api_key: credential.api_key,
      base_url: credential.base_url || null,
      eu_compliant: credential.eu_compliant ?? null, // null = defer to catalog default
    };
    this.backend.save(all);
  }

  removeCredential(id) {
    const all = this.listCredentials();
    delete all[id];
    this.backend.save(all);
  }

  hasCredential(id) {
    return Boolean(this.getCredential(id));
  }

  /**
   * All gateway credentials the user has configured (there can be more than
   * one), each annotated with a resolved `eu_compliant` boolean: the stored
   * override if the user set one, else the catalog's default for that gateway.
   */
  listGatewayCredentials() {
    return Object.entries(this.listCredentials())
      .filter(([, c]) => c.type === "gateway")
      .map(([id, c]) => {
        const catalogEntry = this.gateways.find(g => g.id === id);
        const euCompliant = c.eu_compliant ?? catalogEntry?.eu_compliant ?? false;
        return { id, ...c, eu_compliant: euCompliant };
      });
  }

  /**
   * Resolve which credential to use for a given direct-provider id: prefer a
   * direct key for that exact provider; fall back to any configured gateway
   * (gateways route to most providers through one endpoint).
   *
   * @param {string} providerId
   * @param {object} [opts]
   * @param {boolean} [opts.euOnly] - if true, a gateway is only used as a
   *   fallback when it's marked eu_compliant. A non-compliant gateway sees
   *   every request's content regardless of which model it's routed to, so
   *   silently using one for an EU-only request would leak the request
   *   outside the EU even though the chosen model is EU-hosted. Returns null
   *   rather than a credential in that case — the caller should surface that
   *   a direct key for an EU provider is needed.
   * Returns null if no usable credential exists.
   */
  resolveCredentialForProvider(providerId, opts = {}) {
    const direct = this.getCredential(providerId);
    if (direct && direct.type === "direct") return { source: providerId, ...direct };

    const gateways = this.listGatewayCredentials();
    const usable = opts.euOnly ? gateways.filter(g => g.eu_compliant) : gateways;
    if (usable.length) return { source: usable[0].id, ...usable[0] };

    return null;
  }
}
