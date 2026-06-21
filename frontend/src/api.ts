// api.ts — Typed fetch client for the Talent Terminal API.
//
// Breakdown keys MUST match the _BREAKDOWN_MAP values in api_server.py,
// which in turn mirror the keys produced by ranking_pipeline.py's finalize_ranking().

export interface CandidateMeta {
  current_title: string;
  current_company: string;
  years_exp: number;
  edu_tier_1: boolean;
  open_to_work: boolean;
  willing_to_relocate: boolean;
  github_score: number;     // -1 means not available
  avg_assessment: number;   // -1 means not available
  saved_by_recruiters: number;
  linkedin_connected: boolean;
  verified_email: boolean;
  verified_phone: boolean;
  notice_days: number | null;
  preferred_work_mode: string;
}

/** Keys match _BREAKDOWN_MAP in api_server.py */
export interface CandidateBreakdown {
  semantic: number;
  location: number;
  ml_ratio: number;
  experience: number;
  company: number;
  ml_signals: number;
  behavioral: number;
  recency: number;
  jd_terms: number;
  elite_co: number;
  github: number;
  assessment: number;
}

export interface CandidateResult {
  candidate_id: string;
  rank: number;
  score: number;
  reasoning: string;
  breakdown: CandidateBreakdown;
  meta: CandidateMeta;
}

export interface RankResponse {
  results: CandidateResult[];
  count: number;
}

// Use VITE env variable in prod; fall back to relative /api for unified hosting.
export const API_BASE =
  (import.meta.env?.VITE_API_URL as string | undefined) ?? "/api";

const REQUEST_TIMEOUT_MS = 180_000; // 3 min — cross-encoder can take a while

export async function rankCandidates(
  jd_text: string,
  top_n = 100,
): Promise<RankResponse> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/rank`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ jd_text, top_n }),
      signal: controller.signal,
    });
  } catch (err: unknown) {
    if (err instanceof Error && err.name === "AbortError") {
      throw new Error("Request timed out — the ranking pipeline took too long.", { cause: err });
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }

  if (!res.ok) {
    // Guard against non-JSON error bodies (e.g. plain HTML from unhandled 500)
    const contentType = res.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      const err = await res.json();
      throw new Error(err.detail ?? `Server error ${res.status}`);
    }
    throw new Error(`Server error ${res.status}: ${await res.text()}`);
  }

  return res.json();
}
