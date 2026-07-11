from enum import Enum

class VoiceVariant(str, Enum):
    AMERICAN_FEMALE_WARM = "af_heart"
    AMERICAN_FEMALE_CLEAR = "af_bella"
    AMERICAN_MALE_DEEP = "am_adam"
    AMERICAN_MALE_BALANCED = "am_echo"
    BRITISH_FEMALE = "bf_emma"
    BRITISH_MALE = "bm_daniel"