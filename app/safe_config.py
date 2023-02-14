SAFETY_CONFIG = {
    "NONE": {
        "sld_warmup_steps": 15,
        "sld_guidance_scale": 0,
        "sld_threshold": 0.0,
        "sld_momentum_scale": 0.0,
        "sld_mom_beta": 0.0,
    },
    "WEAK": {
        "sld_warmup_steps": 15,
        "sld_guidance_scale": 20,
        "sld_threshold": 0.0,
        "sld_momentum_scale": 0.0,
        "sld_mom_beta": 0.0,
    },
    "MEDIUM": {
        "sld_warmup_steps": 10,
        "sld_guidance_scale": 1000,
        "sld_threshold": 0.01,
        "sld_momentum_scale": 0.3,
        "sld_mom_beta": 0.4,
    },
    "STRONG": {
        "sld_warmup_steps": 7,
        "sld_guidance_scale": 2000,
        "sld_threshold": 0.025,
        "sld_momentum_scale": 0.5,
        "sld_mom_beta": 0.7,
    },
    "MAX":{
        "sld_warmup_steps": 0,
        "sld_guidance_scale": 5000,
        "sld_threshold": 1.0,
        "sld_momentum_scale": 0.5,
        "sld_mom_beta": 0.7,
    }
}