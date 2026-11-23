import os

import src.config  # noqa: F401  (loads .env)

FLAG = "EDITABLE_BASE"
TOOLS_FLAG = "BUILD_TOOLS"                  # the menu of the construction tools (S11-T03)
READ_ONLY_NOTE = (f"Solo lectura: aquí ves las decisiones de la base. Para cambiarlas, agrega "
                  f"{FLAG}=1 a tu archivo .env y reinicia la web.")


def editable() -> bool:
    return os.getenv(FLAG, "").strip() == "1"


def tools_shown() -> bool:
    """The deployed web shows only the assistant; whoever builds the base opens the tools."""
    return os.getenv(TOOLS_FLAG, "").strip() == "1"
