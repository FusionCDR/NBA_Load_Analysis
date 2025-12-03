
import pandas as pd
from src.model import compare_models, build_model_variants

df = pd.read_parquet('data/cache/features_debug.parquet')

# Build and fit the models
variants = build_model_variants(df)
for name, model in variants.items():
    model.fit(df)  # Fit on full data to get coefficients
# Check coefficients
print("\n" + "="*60)
print("COEFFICIENT SIGNIFICANCE CHECK")
print("="*60)

for name, model in variants.items():
    coeffs = model.get_coefficients()
    significant = coeffs[coeffs['P>|z|'] < 0.05]
    print("SIGNIFICANT PREDICTORS:")
    print(significant[['Coef.', 'P>|z|']].sort_values('P>|z|'))
    print(f"\n{name.upper()} MODEL:")
    travel_cols = [c for c in coeffs.index if 'TRAVEL' in c.upper()]
    
    for col in travel_cols:
        p_val = coeffs.loc[col, 'P>|z|']
        coef = coeffs.loc[col, 'Coef.']
        sig = "!" if p_val < 0.05 else "X"
        print(f"  {col}: {coef:+.4f} (p={p_val:.3f}) {sig}")
        

# Now run the comparison
cv = compare_models(df, n_splits=4, val_size=0.15, test_size=0.15, val_set='val')
print("\nModel Comparison:")
print(cv)


