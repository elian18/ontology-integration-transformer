import os

import src.config  # noqa: F401  (loads .env)

FLAG = "EDITABLE_BASE"
READ_ONLY_NOTE = (f"Solo lectura: aquí ves las decisiones de la base. Para cambiarlas, agrega "
                  f"{FLAG}=1 a tu archivo .env y reinicia la web.")


def editable() -> bool:
    return os.getenv(FLAG, "").strip() == "1"