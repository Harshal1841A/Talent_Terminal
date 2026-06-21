import type { CandidateResult } from "../api";

const BREAKDOWN_COLUMNS: (keyof CandidateResult["breakdown"])[] = [
  "semantic",
  "location",
  "ml_ratio",
  "experience",
  "company",
  "ml_signals",
  "behavioral",
  "recency",
  "jd_terms",
  "elite_co",
  "github",
  "assessment",
];

const COLUMNS = [
  "rank",
  "candidate_id",
  "score",
  "current_title",
  "current_company",
  "years_exp",
  "open_to_work",
  "willing_to_relocate",
  "edu_tier_1",
  "github_score",
  "avg_assessment",
  "saved_by_recruiters",
  "linkedin_connected",
  "verified_email",
  "verified_phone",
  "notice_days",
  "preferred_work_mode",
  ...BREAKDOWN_COLUMNS.map((k) => `signal_${k}`),
  "reasoning",
] as const;

function escapeCsvField(value: unknown): string {
  if (value === null || value === undefined) return "";
  const str = String(value);
  // Quote any field containing a comma, quote, or newline; escape inner quotes.
  if (/[",\n]/.test(str)) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

function candidateToRow(c: CandidateResult): string {
  const values: unknown[] = [
    c.rank,
    c.candidate_id,
    c.score,
    c.meta.current_title,
    c.meta.current_company,
    c.meta.years_exp,
    c.meta.open_to_work,
    c.meta.willing_to_relocate,
    c.meta.edu_tier_1,
    c.meta.github_score,
    c.meta.avg_assessment,
    c.meta.saved_by_recruiters,
    c.meta.linkedin_connected,
    c.meta.verified_email,
    c.meta.verified_phone,
    c.meta.notice_days,
    c.meta.preferred_work_mode,
    ...BREAKDOWN_COLUMNS.map((k) => c.breakdown[k]),
    c.reasoning,
  ];
  return values.map(escapeCsvField).join(",");
}

/**
 * Builds a CSV string from ranked candidates and triggers a browser
 * download. Includes the full signal breakdown and metadata — richer
 * than the slim 4-column submission CSV the backend pipeline writes.
 */
export function downloadResultsAsCsv(
  results: CandidateResult[],
  filename = "talent_terminal_results.csv",
): void {
  if (results.length === 0) return;

  const header = COLUMNS.join(",");
  const rows = results.map(candidateToRow);
  const csvContent = [header, ...rows].join("\r\n");

  // Prepend a UTF-8 BOM so Excel opens the file with correct encoding
  // instead of mangling non-ASCII characters in names/titles.
  const blob = new Blob(["\uFEFF" + csvContent], {
    type: "text/csv;charset=utf-8;",
  });

  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/** Builds a sensible filename including a date stamp. */
export function buildExportFilename(): string {
  const stamp = new Date().toISOString().slice(0, 10);
  return `talent_terminal_results_${stamp}.csv`;
}
