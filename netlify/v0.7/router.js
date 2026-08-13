// Quindral v0.7 demo snapshot — static JS port of classifier.py + router.py +
// explain.py + model_registry.py's filter_by(). Registry data is a frozen
// export (see registry.json / scripts_export_snapshot.py) — this file has
// no network calls, everything runs client-side for a drag-and-drop demo.
// Ported 1:1 from the Python source at v0.7; keep both in sync by hand if
// the Python logic changes and you want a matching demo snapshot.

const ROUTES = ["code", "vision", "long_context", "simple", "reasoning"];

const REFERENCE_EXAMPLES = {
  code: [
    "write a python function that sorts a list",
    "fix this bug in my javascript code",
    "how do I implement a binary search tree",
    "debug this stack trace",
    "why does my css layout keep breaking on mobile",
    "this sql query is returning duplicate rows, what's wrong",
    "my docker container keeps exiting immediately, help",
    "convert this react class component to use hooks",
    "write a shell script to automate this deployment",
    "what's the correct syntax for a git rebase",
  ],
  vision: [
    "what is in this image",
    "describe this photo",
    "analyze the attached screenshot",
    "can you read the text in this scanned document",
    "is there anything unusual in this x-ray",
    "what's wrong with the layout in this UI mockup",
    "identify the objects visible in this picture",
    "transcribe the handwriting shown here",
  ],
  long_context: [
    "summarize this entire document",
    "read through this long report and extract key points",
    "go through this whole contract and flag anything unusual",
    "compare these lengthy research papers for contradictions",
    "extract every action item from this long email thread",
  ],
  reasoning: [
    "explain step by step why this proof works",
    "solve this complex multi-step math problem",
    "plan a strategy considering multiple tradeoffs",
    "work out a word problem involving relative speed and distance",
    "determine whether this argument is logically valid",
    "calculate the probability of a specific card-drawing outcome",
    "figure out the most efficient path between several locations",
  ],
  simple: [
    "what is the capital of france",
    "what time is it in tokyo",
    "define the word ubiquitous",
  ],
};

const STOPWORDS = new Set([
  "a", "an", "the", "is", "in", "of", "this", "what", "how", "do", "i",
  "to", "and", "that", "it", "for", "on", "my", "with", "why",
  "are", "there", "was", "were", "be", "been", "can", "does", "did",
]);

function tokenize(text) {
  const words = text.toLowerCase().match(/[a-z0-9]+/g) || [];
  return words.filter(w => !STOPWORDS.has(w) && w.length > 1);
}

function bagOfWords(text) {
  const bow = new Map();
  for (const w of tokenize(text)) bow.set(w, (bow.get(w) || 0) + 1);
  return bow;
}

function cosine(a, b) {
  let dot = 0;
  for (const [tok, countA] of a) if (b.has(tok)) dot += countA * b.get(tok);
  const magA = Math.sqrt([...a.values()].reduce((s, v) => s + v * v, 0));
  const magB = Math.sqrt([...b.values()].reduce((s, v) => s + v * v, 0));
  if (!magA || !magB) return 0.0;
  return dot / (magA * magB);
}

const REFERENCE_BOWS = Object.fromEntries(
  Object.entries(REFERENCE_EXAMPLES).map(([route, exs]) => [route, exs.map(bagOfWords)])
);

function embeddingSimilarityScores(query) {
  const q = bagOfWords(query);
  const scores = {};
  for (const [route, refBows] of Object.entries(REFERENCE_BOWS)) {
    scores[route] = Math.max(0, ...refBows.map(ref => cosine(q, ref)));
  }
  return scores;
}

function heuristicBoosts(query) {
  const boosts = Object.fromEntries(ROUTES.map(r => [r, 0.0]));
  if (/```|def |function\s*\(|class \w+|import \w+|;\s*$|\b(css|sql|docker|kubernetes|regex|npm|git|api|react|hooks|container|component|query|syntax|compile|stack trace|repo)\b/im.test(query)) boosts.code += 0.3;
  if (/\b(image|photo|picture|screenshot|diagram|x-ray|scan(ned)?|mockup|handwriting)\b/i.test(query)) boosts.vision += 0.3;
  const wordCount = tokenize(query).length;
  if (wordCount > 200) boosts.long_context += 0.3;
  if (/[=+\-*/^]|\b(solve|prove|probability|logically valid|logical fallacy|derive)\b/i.test(query)) boosts.reasoning += 0.15;
  if (wordCount < 12 && query.includes("?")) boosts.simple += 0.15;
  return boosts;
}

// Retrieval-necessity signal: does this query need info newer than any
// model's training cutoff? Orthogonal to the task-type route above.
const CURRENT_INFO_PATTERN = new RegExp(
  "\\b(today|tonight|currently|current|right now|as of|this week|this month|this year|" +
  "latest|breaking|recent(ly)?|up[- ]to[- ]date|" +
  "who (is|are) the (current|latest)|who won|who is winning|" +
  "stock price|exchange rate|weather (in|today|right now)|" +
  "score of|live score|" +
  "20[2-9]\\d)\\b",
  "i"
);

function needsCurrentInfo(query) {
  return CURRENT_INFO_PATTERN.test(query);
}

function classify(query) {
  const simScores = embeddingSimilarityScores(query);
  const boosts = heuristicBoosts(query);
  const scores = Object.fromEntries(ROUTES.map(r => [r, simScores[r] + boosts[r]]));
  const maxScore = Math.max(...Object.values(scores));
  const route = maxScore === 0.0
    ? "simple"
    : Object.entries(scores).reduce((a, b) => (b[1] > a[1] ? b : a))[0];
  return {
    route,
    confidence: Math.round(scores[route] * 1000) / 1000,
    scores,
    needs_current_info: needsCurrentInfo(query),
  };
}

// ---------- registry / hard filters ----------

const DEFAULT_WEIGHTS = { cost: 0.6, env: 0.3, latency: 0.1 };

const ROUTE_CAPABILITY = {
  code: "code",
  vision: "vision",
  long_context: "long_context",
  reasoning: "reasoning",
  simple: "factual",
};

const RELAXATION_ORDER = ["min_context", "web_search", "capability", "eu_only"];
const RELAXED_FILTER_PHRASING = {
  min_context: "the context-length requirement",
  web_search: "the live-search preference",
  capability: "the capability requirement",
  eu_only: "the EU-only filter",
};
const ROUTE_PHRASING = {
  simple: "a simple factual lookup",
  code: "a coding task",
  vision: "a task involving an image",
  long_context: "a task needing a long context window",
  reasoning: "a task needing multi-step reasoning",
};
const CRITERION_PHRASING = { cost: "cost", env: "environmental impact", latency: "response speed" };
const CAPABILITY_PHRASING = {
  code: "generate or debug code",
  vision: "understand images",
  long_context: "handle very long documents",
  reasoning: "work through complex multi-step reasoning",
  factual: "reliably answer general knowledge questions",
  web_search: "search the web for current information",
};

function joinAnd(items) {
  if (items.length === 1) return items[0];
  return `${items.slice(0, -1).join(", ")} or ${items[items.length - 1]}`;
}

function exclusionReasonPhrases(entry) {
  const reasons = [];
  if (entry.missing_capabilities.length) {
    const phrases = entry.missing_capabilities.map(c => CAPABILITY_PHRASING[c] || c);
    reasons.push(`can't ${joinAnd(phrases)}`);
  }
  if (entry.not_eu_hosted) reasons.push("isn't hosted in the EU");
  if (entry.context_deficit) {
    const d = entry.context_deficit;
    reasons.push(`only supports ${d.actual.toLocaleString()} tokens of context, short of the ${d.required.toLocaleString()} this query needs`);
  }
  return reasons;
}

function explainExclusions(excluded) {
  const groups = new Map(); // reason text -> model names
  for (const entry of excluded) {
    const reason = exclusionReasonPhrases(entry).join(" and ") || "was excluded";
    if (!groups.has(reason)) groups.set(reason, []);
    groups.get(reason).push(entry.model);
  }
  const sentences = [];
  for (const [reason, models] of groups) {
    let reasonForGroup = reason;
    if (models.length > 1) {
      reasonForGroup = reasonForGroup.replace("isn't hosted", "aren't hosted").replace("only supports", "only support");
    }
    sentences.push(`${joinAnd(models)} ${reasonForGroup}.`);
  }
  return sentences.join(" ");
}

function filterBy(models, { capability = null, euOnly = false, minContext = 0 } = {}) {
  const required = capability == null ? [] : (typeof capability === "string" ? [capability] : [...capability]);
  return models.filter(m =>
    required.every(c => m.capabilities.includes(c)) &&
    (!euOnly || m.region === "EU") &&
    (m.context_window >= minContext)
  );
}

function normalize(models, keyFn) {
  const values = models.map(keyFn);
  const lo = Math.min(...values), hi = Math.max(...values);
  const out = {};
  for (const m of models) out[m.name] = hi === lo ? 0.0 : (keyFn(m) - lo) / (hi - lo);
  return out;
}

function rank(models, weights) {
  const costNorm = normalize(models, m => m.cost_per_1k_input + m.cost_per_1k_output);
  const envNorm = normalize(models, m => m.energy_wh_per_1k_tokens);
  const latencyNorm = normalize(models, m => m.latency_ms_p50);
  const score = m =>
    (weights.cost || 0) * costNorm[m.name] +
    (weights.env || 0) * envNorm[m.name] +
    (weights.latency || 0) * latencyNorm[m.name];
  return [...models].sort((a, b) => score(a) - score(b));
}

function relaxAndRetry(allModels, requiredCapabilities, euOnly, minContext) {
  const dropped = [];
  for (const filterName of RELAXATION_ORDER) {
    if (filterName === "web_search" && !requiredCapabilities.has("web_search")) continue;
    if (filterName === "capability" && [...requiredCapabilities].filter(c => c !== "web_search").length === 0) continue;
    if (filterName === "eu_only" && !euOnly) continue;
    if (filterName === "min_context" && minContext === 0) continue;
    dropped.push(filterName);

    const trialCapabilities = new Set(requiredCapabilities);
    if (dropped.includes("web_search")) trialCapabilities.delete("web_search");
    if (dropped.includes("capability")) trialCapabilities.clear();
    const trial = {
      capability: trialCapabilities,
      euOnly: dropped.includes("eu_only") ? false : euOnly,
      minContext: dropped.includes("min_context") ? 0 : minContext,
    };
    const survivors = filterBy(allModels, trial);
    if (survivors.length) return { survivors, dropped };
  }
  return { survivors: [], dropped };
}

function route(allModels, query, { weights = DEFAULT_WEIGHTS, euOnly = false, minContext = 0 } = {}) {
  const classification = classify(query);
  const requiredCapabilities = new Set([ROUTE_CAPABILITY[classification.route]]);
  if (classification.needs_current_info) requiredCapabilities.add("web_search");

  let survivors = filterBy(allModels, { capability: requiredCapabilities, euOnly, minContext });
  const excluded = [];
  for (const m of allModels) {
    if (!survivors.includes(m)) {
      excluded.push({
        model: m.name,
        missing_capabilities: [...requiredCapabilities].filter(c => !m.capabilities.includes(c)).sort(),
        not_eu_hosted: Boolean(euOnly && m.region !== "EU"),
        context_deficit: m.context_window < minContext ? { actual: m.context_window, required: minContext } : null,
      });
    }
  }

  let relaxedFilters = [];
  if (!survivors.length) {
    const result = relaxAndRetry(allModels, requiredCapabilities, euOnly, minContext);
    survivors = result.survivors;
    relaxedFilters = result.dropped;
  }

  if (!survivors.length) {
    return {
      classified_as: classification.route,
      eligible_models: [],
      excluded,
      chosen: null,
      binding_criterion: null,
      error: "no candidates survive hard filters, even after relaxing every filter",
    };
  }

  const chosen = rank(survivors, weights)[0];
  const bindingCriterion = Object.entries(weights).reduce((a, b) => (b[1] > a[1] ? b : a))[0];

  const result = {
    classified_as: classification.route,
    eligible_models: survivors.map(m => m.name),
    excluded,
    chosen: chosen.name,
    binding_criterion: bindingCriterion,
  };
  if (relaxedFilters.length) result.relaxed_filters = relaxedFilters;
  return result;
}

function explain(trace, allModels, euOnly = false) {
  const routeDesc = ROUTE_PHRASING[trace.classified_as] || "your request";

  if (trace.chosen === null) {
    return `No model met your requirements for ${routeDesc}, even after relaxing every filter — try loosening your constraints.`;
  }

  const allModelsByName = new Map(allModels.map(m => [m.name, m]));

  const criterionDesc = CRITERION_PHRASING[trace.binding_criterion] || trace.binding_criterion;

  if (trace.relaxed_filters && trace.relaxed_filters.length) {
    const dropped = trace.relaxed_filters.map(f => RELAXED_FILTER_PHRASING[f] || f).join(", ");
    return `No model matched every filter for ${routeDesc}, so we relaxed ${dropped} and suggest ${trace.chosen} as the closest match, still optimizing for ${criterionDesc}.`;
  }

  const sentences = [`Routed to ${trace.chosen} for ${routeDesc}, optimizing primarily for ${criterionDesc}.`];

  const eligible = trace.eligible_models.map(name => allModelsByName.get(name));
  if (eligible.length > 1) {
    const chosen = allModelsByName.get(trace.chosen);
    if (trace.binding_criterion === "env") {
      const costliest = eligible.reduce((a, b) => (b.energy_wh_per_1k_tokens > a.energy_wh_per_1k_tokens ? b : a));
      if (costliest.name !== chosen.name && costliest.energy_wh_per_1k_tokens > 0) {
        const savings = 1 - chosen.energy_wh_per_1k_tokens / costliest.energy_wh_per_1k_tokens;
        if (savings > 0.05) {
          sentences.push(`That's roughly ${Math.round(savings * 100)}% less energy per token than ${costliest.name}, the most energy-intensive model that also qualified.`);
        }
      }
    } else {
      const totalCost = m => m.cost_per_1k_input + m.cost_per_1k_output;
      const costliest = eligible.reduce((a, b) => (totalCost(b) > totalCost(a) ? b : a));
      if (costliest.name !== chosen.name && totalCost(costliest) > 0) {
        const savings = 1 - totalCost(chosen) / totalCost(costliest);
        if (savings > 0.05) {
          sentences.push(`That's roughly ${Math.round(savings * 100)}% cheaper per token than ${costliest.name}, the priciest model that also qualified.`);
        }
      }
    }
  }

  if (trace.excluded.length) sentences.push("Ruled out: " + explainExclusions(trace.excluded));

  if (euOnly) sentences.push(`${trace.chosen} is EU-hosted, meeting your privacy filter.`);

  return sentences.join(" ");
}
