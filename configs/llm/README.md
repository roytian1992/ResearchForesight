# LLM Profiles

This directory stores provider-specific OpenAI-compatible configs and routing policy files.

Recommended usage:
- qwen-235B: harder reasoning, taxonomy expansion, difficult audit decisions
- mimo-v2-flash: cheap batch labeling, candidate polishing, first-pass judges
- mimo-v2-pro: higher-quality rewrite / judge fallback

Use environment variables for secrets instead of hardcoding API keys.
