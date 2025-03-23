from model.view.services import run_pipeline

if __name__ == '__main__':
    file_path = 'model/Watches.csv'   # Update with your actual data path
    target = 'price'        # Update with your target column name    
    model, shap_summary = run_pipeline(file_path, target)
    print("Model training and feature analysis completed.")

    import shap
    shap.summary_plot(shap_summary.values, shap_summary.data)
    