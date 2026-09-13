cd dark-octet-dm
pip install -e ".[dev]"
pytest tests/ -v --tb=short
python -m dark_octet.spectrum.gmo_formula   # prints mass table
python -m dark_octet.sommerfeld.enhancement # prints σ_T table + saves PDF
docker compose -f docker/docker-compose.yml up  # full JupyterLab