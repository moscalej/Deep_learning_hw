from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier



results_rf = pd.DataFrame(columns=[100, 1_000, 10_000])

clf_rf = RandomForestClassifier()
clf_svm = SVC()

for train_size in results_rf.columns:
    for n_neighbors in tqdm(range(10, 60, 10)):
        X_train_small, _, y_train_small, _ = train_test_split(
            x_train, y_train, train_size=train_size, test_size=0.0,
            random_state=42)

        y_train_small = np.argmax(y_train_small, 1)

        VGG_x_train_small = clf.predict(X_train_small,normalize=False)

        neigh = RandomForestClassifier(n_estimators=n_neighbors, n_jobs=-1)
        neigh.fit(VGG_x_train_small, y_train_small)
        results_rf.loc[n_neighbors, train_size] = neigh.score(VGG_x_test, y_test)