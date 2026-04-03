import os

from hypothesis import settings

settings.register_profile("long", max_examples=50000, deadline=None)
settings.register_profile("ci", max_examples=500, deadline=None)
settings.register_profile("dev", max_examples=50, deadline=None)
settings.register_profile("debug", max_examples=10, deadline=None)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
