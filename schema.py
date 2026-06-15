from typing import Optional
from pydantic import BaseModel, Field

class CandidateMetadata(BaseModel):
    candidate_id: str
    current_title: str
    current_company: str
    years_exp: float
    honeypot: bool
    has_product_company: bool
    consulting_only: bool
    wrong_title: bool
    ml_signal_count: float
    jd_term_bonus: float
    has_external_validation: bool
    elite_company_bonus: float
    edu_tier_1: bool
    response_rate: float
    notice_days: int
    open_to_work: bool
    github_score: float = Field(default=-1, description="GitHub activity score")
    last_active_days: int
    interview_completion: float
    offer_acceptance: float
    profile_completeness: float = Field(default=0, description="Profile completeness percentage")
    avg_assessment: float
    core_skill_score: float
    willing_to_relocate: bool
    preferred_work_mode: str
    saved_by_recruiters: int
    applications_submitted: int
    search_appearance_30d: int
    avg_response_time_hours: float
    endorsements_received: int
    connection_count: int
    linkedin_connected: bool
    verified_email: bool
    verified_phone: bool
    research_founding_score: float
    skill_count: int
    location_score: float
    location_score_city: str
    ml_role_ratio: float
    ml_role_months: int
    total_months: int
    title_chaser: bool
    avg_tenure_months: float
    doc_text: Optional[str] = None
    lgbm_score: Optional[float] = None
    
    # Allows additional config parameters dynamically appended in app.py or rank.py
    model_config = {
        "extra": "allow"
    }
