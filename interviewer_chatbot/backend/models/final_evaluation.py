from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class FinalEvaluation:
    overall_quality: int
    strengths: List[str]
    areas_for_improvement: List[str]
    recommendation: str
    final_feedback: str

    def model_dump(self) -> Dict[str, Any]:
        return {
            "overall_quality": self.overall_quality,
            "strengths": self.strengths,
            "areas_for_improvement": self.areas_for_improvement,
            "recommendation": self.recommendation,
            "final_feedback": self.final_feedback,
        }
