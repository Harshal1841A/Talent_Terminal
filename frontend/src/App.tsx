import { useState } from 'react';
import { motion } from 'framer-motion';
import { Download } from 'lucide-react';
import { rankCandidates } from './api';
import type { CandidateResult } from './api';
import { CandidateRow } from './components/CandidateRow';
import { ComparisonPanel } from './components/ComparisonPanel';
import { downloadResultsAsCsv, buildExportFilename } from './utils/exportCsv';

function App() {
  const [jdText, setJdText] = useState('');
  const [results, setResults] = useState<CandidateResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);

  const handleRank = async () => {
    if (!jdText.trim() || loading) return;
    setLoading(true);
    setError(null);
    try {
      const data = await rankCandidates(jdText);
      setResults(data.results);
      setSelectedIds([]);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to rank candidates');
    } finally {
      setLoading(false);
    }
  };

  const toggleSelection = (id: string) => {
    setSelectedIds(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id],
    );
  };

  const handleDownload = () => {
    downloadResultsAsCsv(results, buildExportFilename());
  };

  const selectedCandidates = results.filter(c => selectedIds.includes(c.candidate_id));

  return (
    <div className="min-h-screen relative pb-40 selection:bg-[var(--accent)] selection:text-white">
      <header className="px-6 py-6 border-b border-[var(--border)] glass-panel sticky top-0 z-10">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-[var(--accent)] to-[var(--text-primary)] bg-clip-text text-transparent">
            Talent Terminal
          </h1>
          <p className="font-mono text-xs text-[var(--text-tertiary)] flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-[var(--accent)] shadow-[0_0_8px_var(--accent)] animate-pulse"></span>
            Hybrid Retrieval • Cross-Encoder Re-Ranking
          </p>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8">
        <section className="mb-8 glass-panel p-6 rounded-xl">
          <label className="block text-xs font-bold tracking-widest uppercase text-[var(--accent)] mb-4 flex items-center gap-2">
            Job Description Input
          </label>
          <textarea
            value={jdText}
            onChange={(e) => setJdText(e.target.value)}
            onKeyDown={e => { if (e.ctrlKey && e.key === 'Enter') handleRank(); }}
            rows={4}
            placeholder="Paste a job description to rank candidates..."
            className="w-full rounded-lg bg-[rgba(0,0,0,0.2)] border border-[var(--border)] p-4 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--border-focus)] focus:ring-1 focus:ring-[var(--border-focus)] font-mono transition-all resize-y"
          />
          <div className="flex items-center justify-between mt-4">
            <span className="text-xs text-[var(--text-tertiary)] font-mono bg-[rgba(255,255,255,0.05)] px-2 py-1 rounded">
              Ctrl+Enter to submit
            </span>
            <button
              onClick={handleRank}
              disabled={loading}
              className="rounded-lg bg-gradient-to-r from-[var(--accent)] to-[var(--accent-amber)] px-6 py-2.5 text-sm font-semibold text-white hover:opacity-90 disabled:opacity-50 transition-all shadow-[0_0_15px_rgba(6,182,212,0.3)] hover:shadow-[0_0_20px_rgba(6,182,212,0.5)]"
            >
              {loading ? 'Ranking...' : 'Rank Candidates'}
            </button>
          </div>
          {error && <p className="mt-4 text-sm text-red-400 font-medium">{error}</p>}
        </section>

        {results.length > 0 && (
          <>
            <div className="mb-3 flex items-center justify-between">
              <p className="text-xs text-[var(--text-tertiary)]">
                {results.length} candidate{results.length === 1 ? '' : 's'} ranked
              </p>
              <button
                onClick={handleDownload}
                className="inline-flex items-center gap-1.5 rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-1.5 text-xs font-medium text-[var(--text-secondary)] hover:bg-[var(--surface-hover)] hover:text-[var(--text-primary)]"
              >
                <Download size={13} />
                Download CSV
              </button>
            </div>
            <motion.div
              initial="hidden"
              animate="visible"
              variants={{
                hidden: {},
                visible: { transition: { staggerChildren: 0.03 } },
              }}
              className="flex flex-col gap-px rounded-md border border-[var(--border)] overflow-hidden"
            >
            {results.map((c) => (
              <CandidateRow
                key={c.candidate_id}
                candidate={c}
                selected={selectedIds.includes(c.candidate_id)}
                onToggleSelect={toggleSelection}
              />
            ))}
            </motion.div>
          </>
        )}
      </main>

      <ComparisonPanel
        candidates={selectedCandidates}
        onClose={() => setSelectedIds([])}
        onRemove={toggleSelection}
      />
    </div>
  );
}

export default App;
