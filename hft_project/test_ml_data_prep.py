import json
from src.data.universe_manager import UniverseManager
from src.research.alpha_discovery import AlphaDiscovery

um = UniverseManager('dummy')
data = um.load_universe_data('CSI300', '20230101', '20230131')  # Use 1 month for quick test

with open('config/strategies/momentum.json') as f:
    base_config = json.load(f)

ad = AlphaDiscovery(um, base_config)
X, y = ad._prepare_ml_data(data, 1)

print('X shape:', X.shape)
print('y shape:', y.shape)
print(X.head())
print(y.head())

if X.empty or y.empty:
    raise RuntimeError('ML data preparation failed: X or y is empty!') 