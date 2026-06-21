import React from 'react';
import type { CandidateBreakdown } from '../api';

interface ScoreBreakdownProps {
  breakdown: CandidateBreakdown;
}

const BREAKDOWN_META: { key: keyof CandidateBreakdown; label: string }[] = [
  { key: 'semantic',    label: 'SEM' }, 
  { key: 'location',    label: 'LOC' }, 
  { key: 'ml_ratio',    label: 'MLR' }, 
  { key: 'experience',  label: 'EXP' }, 
  { key: 'company',     label: 'CO'  }, 
  { key: 'ml_signals',  label: 'MLS' }, 
  { key: 'behavioral',  label: 'BEH' }, 
  { key: 'recency',     label: 'REC' }, 
  { key: 'jd_terms',    label: 'JD'  }, 
  { key: 'elite_co',    label: 'ELT' }, 
  { key: 'github',      label: 'GH'  }, 
  { key: 'assessment',  label: 'ASM' }, 
];

export const ScoreBreakdown: React.FC<ScoreBreakdownProps> = ({ breakdown }) => {
  return (
    <div className="grid grid-cols-4 md:grid-cols-6 gap-y-3 gap-x-2 mt-4 p-4 bg-[var(--bg)] border border-[var(--border)] rounded">
      {BREAKDOWN_META.map(({ key, label }) => {
        const val = breakdown[key] ?? 0;
        const isHigh = val > 0.95;
        const isMid = val > 0.5 && !isHigh;
        const colorClass = isHigh ? 'text-[var(--accent-amber)] font-medium' : (isMid ? 'text-[var(--accent)] font-normal' : 'text-[var(--text-tertiary)] font-normal');
        const borderClass = isHigh ? 'border-[var(--accent-amber)]' : 'border-[var(--border)]';
        
        return (
          <div key={key} className={`flex flex-col border-l-[2px] ${borderClass} pl-2`}>
            <span className="text-[10px] font-bold tracking-wider text-[var(--text-secondary)] mb-0.5">{label}</span>
            <span className={`font-mono text-[13px] ${colorClass}`}>
              {val.toFixed(3)}
            </span>
          </div>
        );
      })}
    </div>
  );
};
