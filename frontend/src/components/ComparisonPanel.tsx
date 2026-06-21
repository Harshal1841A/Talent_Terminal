import React from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { X, Check } from 'lucide-react';
import type { CandidateResult } from '../api';

interface ComparisonPanelProps {
  candidates: CandidateResult[];
  onClose: () => void;
  onRemove: (id: string) => void;
}

export const ComparisonPanel: React.FC<ComparisonPanelProps> = ({ candidates, onClose, onRemove }) => (
  <AnimatePresence>
    {candidates.length > 0 && (
      <motion.div
        key="comparison-panel"
        initial={{ y: '100%' }}
        animate={{ y: 0 }}
        exit={{ y: '100%' }}
        transition={{ type: 'spring', damping: 28, stiffness: 220 }}
        className="fixed bottom-0 left-0 right-0 z-50 glass-panel border-t border-[var(--border)] border-l-0 border-r-0 border-b-0 p-6 pb-8 shadow-[0_-10px_40px_rgba(0,0,0,0.5)]"
      >
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--text-primary)] flex items-center gap-3">
              Comparing {candidates.length} candidate{candidates.length !== 1 ? 's' : ''}
              {candidates.length > 3 && (
                <span className="text-[10px] font-mono text-white bg-[var(--accent-amber)] px-2 py-0.5 rounded-sm">
                  Best with ≤ 3
                </span>
              )}
            </h3>
            <button
              onClick={onClose}
              aria-label="Close comparison panel"
              className="p-1.5 border border-transparent hover:border-[var(--border)] hover:bg-[var(--bg)] rounded text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
            >
              <X size={16} />
            </button>
          </div>

          <div className="flex gap-4 overflow-x-auto pb-4 snap-x">
            {candidates.map(c => (
              <motion.div
                layout
                key={c.candidate_id}
                className="min-w-[280px] flex-1 glass-row rounded-xl p-5 relative group snap-start"
              >
                <button
                  onClick={() => onRemove(c.candidate_id)}
                  aria-label={`Remove ${c.candidate_id} from comparison`}
                  className="absolute top-3 right-3 p-1.5 bg-[var(--surface)] border border-[var(--border)] rounded opacity-0 group-hover:opacity-100 transition-opacity text-[var(--text-secondary)] hover:text-[var(--score-low)] hover:border-[var(--score-low)]"
                >
                  <X size={12} />
                </button>

                <h4 className="font-semibold text-[15px] text-[var(--text-primary)] mb-0.5 pr-8 truncate tracking-tight" title={c.candidate_id}>
                  {c.meta.current_title || c.candidate_id}
                </h4>
                {c.meta.current_company && (
                  <p className="text-xs font-medium text-[var(--text-secondary)] mb-4 truncate">{c.meta.current_company}</p>
                )}
                <div className="text-3xl font-mono text-[var(--accent)] mb-6">{c.score.toFixed(2)}</div>

                <div className="space-y-3 text-[10px] font-bold uppercase tracking-widest">
                  <div className="flex justify-between border-b border-[var(--border)] pb-2">
                    <span className="text-[var(--text-secondary)]">Experience</span>
                    <span className="text-[var(--text-primary)] font-mono">{c.meta.years_exp} yrs</span>
                  </div>
                  <div className="flex justify-between border-b border-[var(--border)] pb-2">
                    <span className="text-[var(--text-secondary)]">Tier 1 Edu</span>
                    <span className="text-[var(--text-primary)]">
                      {c.meta.edu_tier_1
                        ? <Check size={14} className="text-[var(--score-high)]" />
                        : <span className="text-[var(--text-tertiary)]">—</span>}
                    </span>
                  </div>
                  <div className="flex justify-between border-b border-[var(--border)] pb-2">
                    <span className="text-[var(--text-secondary)]">GitHub</span>
                    <span className="text-[var(--text-primary)] font-mono">
                      {c.meta.github_score > 0 ? c.meta.github_score.toFixed(0) : <span className="text-[var(--text-tertiary)]">—</span>}
                    </span>
                  </div>
                  <div className="flex justify-between pt-1">
                    <span className="text-[var(--text-secondary)]">Assessment</span>
                    <span className="text-[var(--text-primary)] font-mono">
                      {c.meta.avg_assessment > 0 ? c.meta.avg_assessment.toFixed(1) : <span className="text-[var(--text-tertiary)]">—</span>}
                    </span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>
    )}
  </AnimatePresence>
);
