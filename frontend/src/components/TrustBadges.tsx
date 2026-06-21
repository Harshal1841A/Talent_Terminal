import { Mail, Phone, Link, GraduationCap } from 'lucide-react';
import type { CandidateMeta } from '../api';

export function TrustBadges({ meta }: { meta: CandidateMeta }) {
  const badges = [
    meta.verified_email && { icon: Mail, label: 'Email verified' },
    meta.verified_phone && { icon: Phone, label: 'Phone verified' },
    meta.linkedin_connected && { icon: Link, label: 'LinkedIn connected' },
    meta.edu_tier_1 && { icon: GraduationCap, label: 'Tier-1 education' },
  ].filter(Boolean) as { icon: typeof Mail; label: string }[];

  return (
    <div className="flex gap-1.5">
      {badges.map(({ icon: Icon, label }) => (
        <span key={label} title={label} className="text-[var(--text-tertiary)] flex items-center">
          <Icon size={13} />
        </span>
      ))}
    </div>
  );
}
