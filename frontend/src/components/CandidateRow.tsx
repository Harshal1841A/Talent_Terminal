import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { Variants } from 'framer-motion';
import { ChevronDown } from 'lucide-react';
import type { CandidateResult } from '../api';
import { ScoreBreakdown } from './ScoreBreakdown';
import { TrustBadges } from './TrustBadges';

interface Props {
  candidate: CandidateResult;
  selected: boolean;
  onToggleSelect: (id: string) => void;
}

const rowVariants: Variants = {
  hidden: { opacity: 0, x: -10 },
  visible: { opacity: 1, x: 0, transition: { duration: 0.3, ease: 'easeOut' } },
};

function scoreColor(score: number, maxScore: number): string {
  const ratio = score / maxScore;
  if (ratio > 0.8) return 'var(--score-high)'; // Amber
  if (ratio < 0.4) return 'var(--score-low)';  // Muted Slate
  return 'var(--score-mid)';                   // Cyan
}

export function CandidateRow({ candidate, selected, onToggleSelect }: Props) {
  const [expanded, setExpanded] = useState(false);

  return (
    <motion.div
      layout
      variants={rowVariants}
      className={`glass-row transition-all rounded-lg mb-2 overflow-hidden ${
        selected ? 'border-[color:var(--accent)] shadow-[0_0_15px_rgba(6,182,212,0.2)]' : ''
      }`}
    >
      <div className="flex items-center gap-4 px-5 py-4">
        <input
          type="checkbox"
          checked={selected}
          onChange={() => onToggleSelect(candidate.candidate_id)}
          className="h-4 w-4 accent-[var(--accent)] cursor-pointer"
        />
        <span className="w-8 text-[13px] font-mono font-medium text-[var(--text-tertiary)]">
          {candidate.rank}
        </span>
        <div className="flex-1 min-w-0">
          <p className="truncate text-[15px] font-medium text-[var(--text-primary)]">
            {candidate.meta.current_title || 'Untitled role'}
            {candidate.meta.current_company && (
              <span className="text-[var(--text-secondary)] font-normal">
                {' '}@ {candidate.meta.current_company}
              </span>
            )}
          </p>
        </div>
        <TrustBadges meta={candidate.meta} />
        <span
          className="font-mono text-base font-bold tabular-nums px-3 py-1 rounded bg-[rgba(0,0,0,0.3)] shadow-inner"
          style={{ color: scoreColor(candidate.score, 220) }}
        >
          {(candidate.score).toFixed(1)} <span className="text-[10px] text-[var(--text-tertiary)] opacity-70">/ 220</span>
        </span>
        <button
          onClick={() => setExpanded((e) => !e)}
          aria-label="Show ranking detail"
          className="text-[var(--text-tertiary)] hover:text-[var(--text-primary)] p-1 transition-colors outline-none focus:ring-1 focus:ring-[var(--accent)] rounded"
        >
          <motion.span animate={{ rotate: expanded ? 180 : 0 }} className="block" transition={{ duration: 0.2 }}>
            <ChevronDown size={16} strokeWidth={2.5} />
          </motion.span>
        </button>
      </div>

      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: 'easeInOut' }}
            className="overflow-hidden"
          >
            <div className="border-t border-[var(--border)] px-6 py-5 bg-[rgba(0,0,0,0.15)] shadow-inner">
              <p className="text-[13px] leading-relaxed text-[var(--text-secondary)] mb-5 max-w-4xl border-l-2 border-[var(--accent)] pl-4">
                {candidate.reasoning}
              </p>
              <ScoreBreakdown breakdown={candidate.breakdown} />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
