def run(in_csv='data/voice_features.csv', test_size=0.2, random_state=42):
    df = pd.read_csv(in_csv)

    
    if 'filename' in df.columns:
        df = df.drop(columns=['filename'])

    df = df.dropna()  # simple cleaning

    if 'label' in df.columns:
        y = df['label']
        X = df.drop(columns=['label'])
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        joblib.dump(le, 'models/label_encoder.joblib')
    else:
        raise ValueError('CSV must contain `label` column')

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )

    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'models/scaler.joblib')

    
    import os
    os.makedirs('data/processed', exist_ok=True)
    np.savez('data/processed/train_test.npz',
             X_train=X_train_scaled, X_test=X_test_scaled, y_train=y_train, y_test=y_test)
    print('Saved processed arrays and scaler')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='in_csv', default='data/voice_features.csv')
    args = parser.parse_args()
    run(args.in_csv)
