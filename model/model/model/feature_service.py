from model.model.powershap_adapter import calculate_shap_values


def analyze_features(model, data, target: str):
    """
    Analisa as features utilizando SHAP e retorna um resumo dos valores.
    """
    shap_summary = calculate_shap_values(model, data, target)
    return shap_summary
