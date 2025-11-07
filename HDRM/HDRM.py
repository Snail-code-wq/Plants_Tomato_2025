import numpy as np
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

# ---------------------------
# 
# ---------------------------
depth_camera = np.array([
                         ])
depth_groundtruth = np.array([
                              ])


assert depth_camera.shape == depth_groundtruth.shape, 

# ---------------------------
# 
# ---------------------------
TEST_RATIO = 0.1       # 9:1 
RANDOM_STATE = 42
CV_FOLDS = 10           # 

# ---------------------------
# 
# ---------------------------
def simplified_error_model(d, a, b, f, g):
    return (1 + a) * d + b * d**2 + f + g / d

# ---------------------------
# 
# ---------------------------
def make_features(d_arr):
    d = d_arr.reshape(-1, 1)
    return np.column_stack([d, d**2, d**3, np.log(d + 1), 1.0 / d])

# --------------------------
# 
# ---------------------------
X = depth_camera.reshape(-1, 1)
y = depth_groundtruth

X_train_cam, X_test_cam, y_train, y_test = train_test_split(
    X, y, test_size=TEST_RATIO, random_state=RANDOM_STATE, shuffle=True
)

# ---------------------------
# 
# ---------------------------
X_train_feats = make_features(X_train_cam.flatten())
X_test_feats = make_features(X_test_cam.flatten())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feats)
X_test_scaled = scaler.transform(X_test_feats)

# ============================================================
#
# ============================================================
params, _ = curve_fit(simplified_error_model, X_train_cam.flatten(), y_train)
phys_pred_train = simplified_error_model(X_train_cam.flatten(), *params)
residual_train = y_train - phys_pred_train

rf_hybrid = RandomForestRegressor(max_depth=3, n_estimators=27, random_state=RANDOM_STATE)
rf_hybrid.fit(X_train_scaled, residual_train)

# 
phys_pred_test = simplified_error_model(X_test_cam.flatten(), *params)
residual_pred_test = rf_hybrid.predict(X_test_scaled)
hybrid_pred_test = phys_pred_test + residual_pred_test

rmse_hybrid_train = np.sqrt(mean_squared_error(y_train, phys_pred_train + rf_hybrid.predict(X_train_scaled)))
rmse_hybrid_test = np.sqrt(mean_squared_error(y_test, hybrid_pred_test))

# 
cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
neg_mse_scores = cross_val_score(rf_hybrid, X_train_scaled, residual_train, scoring='neg_mean_squared_error', cv=cv)
cv_rmse_hybrid = np.sqrt(-neg_mse_scores).mean()

# 
cv_test = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
rmse_scores_test_hybrid = []

for train_idx, val_idx in cv_test.split(X_test_scaled):
    X_val_train, X_val_test = X_test_scaled[train_idx], X_test_scaled[val_idx]
    y_val_train, y_val_test = y_test[train_idx], y_test[val_idx]
    d_val_train = X_test_cam.flatten()[train_idx]
    d_val_test = X_test_cam.flatten()[val_idx]

    params_val, _ = curve_fit(simplified_error_model, d_val_train, y_val_train)
    phys_pred_val_train = simplified_error_model(d_val_train, *params_val)
    residual_val_train = y_val_train - phys_pred_val_train

    rf_val = RandomForestRegressor(max_depth=5, n_estimators=27, random_state=RANDOM_STATE)
    rf_val.fit(X_val_train, residual_val_train)

    phys_pred_val_test = simplified_error_model(d_val_test, *params_val)
    residual_pred_val_test = rf_val.predict(X_val_test)
    hybrid_pred_val_test = phys_pred_val_test + residual_pred_val_test

    rmse_val = np.sqrt(mean_squared_error(y_val_test, hybrid_pred_val_test))
    rmse_scores_test_hybrid.append(rmse_val)

cv_rmse_hybrid_test = np.mean(rmse_scores_test_hybrid)

# ============================================================
# 
# ============================================================
phys_only_pred_train = simplified_error_model(X_train_cam.flatten(), *params)
phys_only_pred_test = simplified_error_model(X_test_cam.flatten(), *params)
rmse_phys_only_train = np.sqrt(mean_squared_error(y_train, phys_only_pred_train))
rmse_phys_only_test = np.sqrt(mean_squared_error(y_test, phys_only_pred_test))

cv_phys = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
rmse_phys_cv_scores = []
for train_idx, val_idx in cv_phys.split(X):
    d_train, d_val = X[train_idx].flatten(), X[val_idx].flatten()
    y_train_cv, y_val_cv = y[train_idx], y[val_idx]
    try:
        params_cv, _ = curve_fit(simplified_error_model, d_train, y_train_cv)
        y_val_pred = simplified_error_model(d_val, *params_cv)
        rmse_fold = np.sqrt(mean_squared_error(y_val_cv, y_val_pred))
        rmse_phys_cv_scores.append(rmse_fold)
    except RuntimeError:
        continue
cv_rmse_phys_only = np.mean(rmse_phys_cv_scores)

# ============================================================
# 
# ============================================================
rf_only = RandomForestRegressor(max_depth=20, n_estimators=55, random_state=RANDOM_STATE)
rf_only.fit(X_train_scaled, y_train)
rf_only_pred_train = rf_only.predict(X_train_scaled)
rf_only_pred_test = rf_only.predict(X_test_scaled)

rmse_rf_only_train = np.sqrt(mean_squared_error(y_train, rf_only_pred_train))
rmse_rf_only_test = np.sqrt(mean_squared_error(y_test, rf_only_pred_test))

neg_mse_scores_rf_only = cross_val_score(rf_only, X_train_scaled, y_train, scoring='neg_mean_squared_error', cv=cv)
cv_rmse_rf_only = np.sqrt(-neg_mse_scores_rf_only).mean()

cv_test_rf_only = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
rmse_scores_test_rf_only = []

for train_idx, val_idx in cv_test_rf_only.split(X_test_scaled):
    X_val_train, X_val_test = X_test_scaled[train_idx], X_test_scaled[val_idx]
    y_val_train, y_val_test = y_test[train_idx], y_test[val_idx]
    rf_temp = RandomForestRegressor(max_depth=10, n_estimators=75, random_state=RANDOM_STATE)
    rf_temp.fit(X_val_train, y_val_train)
    y_val_pred = rf_temp.predict(X_val_test)
    rmse_val = np.sqrt(mean_squared_error(y_val_test, y_val_pred))
    rmse_scores_test_rf_only.append(rmse_val)

cv_rmse_rf_only_test = np.mean(rmse_scores_test_rf_only)

# ============================================================
# 
# ============================================================
print("==== Results (9:1 split, CV folds = 10) ====")
print("Hybrid (phys + RF resid):")
print(f"  Train RMSE: {rmse_hybrid_train:.3f} mm")
print(f"  Test  RMSE: {rmse_hybrid_test:.3f} mm")
print(f"  CV (train only) RMSE: {cv_rmse_hybrid:.3f} mm")
print(f"  CV (test 10-fold) RMSE: {cv_rmse_hybrid_test:.3f} mm\n")

print("Physical-only (simplified model only):")
print(f"  Train RMSE: {rmse_phys_only_train:.3f} mm")
print(f"  Test  RMSE: {rmse_phys_only_test:.3f} mm")
print(f"  10-fold CV RMSE: {cv_rmse_phys_only:.3f} mm\n")

print("RF-only (direct prediction):")
print(f"  Train RMSE: {rmse_rf_only_train:.3f} mm")
print(f"  Test  RMSE: {rmse_rf_only_test:.3f} mm")
print(f"  CV (train only) RMSE: {cv_rmse_rf_only:.3f} mm")
print(f"  CV (test 10-fold) RMSE: {cv_rmse_rf_only_test:.3f} mm\n")

# ============================================================
# 
# ============================================================
plt.figure(figsize=(8,6))
plt.scatter(X_test_cam.flatten(), y_test - hybrid_pred_test, label='Hybrid Residuals', alpha=0.6)
plt.scatter(X_test_cam.flatten(), y_test - phys_only_pred_test, label='Phys-only Residuals', alpha=0.6)
plt.scatter(X_test_cam.flatten(), y_test - rf_only_pred_test, label='RF-only Residuals', alpha=0.6)
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Depth Camera (mm)")
plt.ylabel("Residual (groundtruth - pred) (mm)")
plt.title("Test Residuals Comparison")
plt.legend()
plt.show()

# ============================================================
# 
# ============================================================
joblib.dump(scaler, 'scaler_hdrm.pkl')
joblib.dump(rf_hybrid, 'rf_hybrid_residuals.pkl')
joblib.dump(rf_only, 'rf_only_direct.pkl')
np.save('params_physical_train.npy', params)

print("Models and params saved.")
