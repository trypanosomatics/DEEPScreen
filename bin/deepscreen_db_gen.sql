CREATE TABLE trained_models (
    target_id VARCHAR(255),
    trained_model_matrix MEDIUMBLOB,
    trained_model_matrix_path MEDIUMTEXT,
    precision FLOAT(255,10),
    recall FLOAT(255,10),
    f1_score FLOAT(255,10),
    accuracy FLOAT(255,10),
    mcc FLOAT(255,10),
    true_positive MEDIUMINT(255),
    false_positive MEDIUMINT(255),
    true_negative MEDIUMINT(255),
    false_negative MEDIUMINT(255)
)