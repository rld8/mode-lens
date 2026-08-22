# ModeLens — guía en español

ModeLens convierte un vídeo corto de una pieza flexible en un experimento modal
auditable. No es una herramienta de inspección estructural: sirve para aprender,
comparar ensayos controlados y demostrar un pipeline científico completo.

## Inicio rápido

```bash
uv sync --all-groups
uv run python scripts/build_demo_assets.py
uv run modelens demo --output runs/demo
make demo
```

El comando genera `result.json`, `signals.csv`, `trajectories.csv`, `arrays.npz`,
`tracking_overlay.mp4` y un informe HTML autocontenido. La interfaz permite revisar la
calidad, las trayectorias, la PSD, las formas modales, el ajuste físico y una comparación.

## Cómo interpretar los resultados

- La frecuencia indica cuántos ciclos por segundo tiene un patrón dominante.
- El damping solo aparece cuando al menos cinco picos siguen un decaimiento
  aproximadamente exponencial; si no, se marca como no identificable.
- MAC próximo a uno significa formas semejantes, no que la pieza sea segura.
- Con frecuencias únicamente se ajusta `EI/(rho*A)`. Separar `E`, geometría y densidad
  exige mediciones externas.

Consulta `experiment_protocol.md`, `mathematical_model.md` y `limitations.md` antes de
grabar un ensayo real.

