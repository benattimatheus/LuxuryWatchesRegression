import shap


def calculate_shap_values(model, data, target: str):
    """
    Calcula os valores SHAP para as features usando o modelo treinado.
    
    Parâmetros:
      - model: Modelo treinado (deve ter um método predict ou similar).
      - data: DataFrame com os dados completos.
      - target: Nome da coluna alvo (a ser removida para obter X).
    
    Retorna:
      - Objeto com os valores SHAP.
    """
    # Separa as features (X) do target (y)
    X = data.drop(columns=[target])
    
    # Utiliza um explicador apropriado para o modelo. 
    # Aqui, usaremos o KernelExplainer para modelos que não são baseados em árvore,
    # mas se o modelo for baseado em árvore, pode-se usar TreeExplainer.
    explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)
    return shap_values
