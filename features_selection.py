def select_features(X, y, k=20):
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X, y)
    mask = selector.get_support()
    selected_idx = np.where(mask)[0].tolist()
    return selected_idx, selector


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_csv', default='data/voice_features.csv')
    parser.add_argument('--k', type=int, default=20)
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)
    labels = df['label']
    X = df.drop(columns=['label','filename'] if 'filename' in df.columns else ['label'])
    le = joblib.load('models/label_encoder.joblib')
    y = le.transform(labels)

    selected_idx, selector = select_features(X, y, k=args.k)
    selected_columns = X.columns[selected_idx].tolist()
    print('Selected:', selected_columns)

    
    joblib.dump(selector, 'models/selector.joblib')
    with open('models/selector_columns.json', 'w') as f:
        json.dump(selected_columns, f, indent=2)

    print('Saved selector andolumn list')
