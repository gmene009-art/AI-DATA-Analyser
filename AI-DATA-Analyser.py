import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class SmartDataAnalyzer:
    def __init__(self, filepath):
        """Load data from CSV"""
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        self.original_df = self.df.copy()
        self.anomalies = None
        self.predictions = None
        self.insights = []
        print(f"[✓] Data loaded: {self.df.shape[0]} rows and {self.df.shape[1]} columns")
    
    def clean_data(self, auto_suggest=True):
        """
        Clean data with intelligent suggestions
        """
        cleaning_suggestions = []
        
        # Delete completely empty columns
        empty_cols = self.df.columns[self.df.isnull().all()].tolist()
        if empty_cols:
            cleaning_suggestions.append(f"🗑️ Empty columns: {empty_cols}")
            self.df.drop(columns=empty_cols, inplace=True)
        
        # Delete duplicate rows
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            cleaning_suggestions.append(f"🔄 Duplicate rows: {duplicates}")
            self.df.drop_duplicates(inplace=True)
        
        # Handle missing values
        for col in self.df.columns:
            missing = self.df[col].isnull().sum()
            if missing > 0:
                if self.df[col].dtype in [np.float64, np.int64]:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                    cleaning_suggestions.append(f"📊 {col}: Replaced {missing} values with median")
                else:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                    cleaning_suggestions.append(f"📝 {col}: Replaced {missing} values with most frequent value")
        
        if auto_suggest:
            print("\n[🧹 Cleaning Suggestions]")
            for suggestion in cleaning_suggestions:
                print(f"  {suggestion}")
        
        print("[✓] Data cleaned successfully")
        return cleaning_suggestions
    
    def basic_statistics(self):
        """Basic statistics with intelligent insights"""
        stats = self.df.describe()
        print("\n[📊 Basic Statistics]")
        print(stats)
        
        # Add insights
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            std = self.df[col].std()
            mean = self.df[col].mean()
            if std > mean * 0.5:
                self.insights.append(f"⚠️ {col}: High variance (std = {std:.2f})")
        
        return stats
    
    def detect_anomalies(self, contamination=0.1):
        """
        Detect anomalies using Isolation Forest
        """
        numeric_data = self.df.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            print("[⚠️] Not enough columns for anomaly detection")
            return
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(numeric_data)
        
        self.df['is_anomaly'] = anomaly_labels
        anomaly_count = (anomaly_labels == -1).sum()
        self.anomalies = self.df[self.df['is_anomaly'] == -1]
        
        print(f"\n[🔍 Anomaly Detection]")
        print(f"  Detected {anomaly_count} anomalies ({anomaly_count/len(self.df)*100:.2f}%)")
        
        if anomaly_count > 0:
            self.insights.append(f"🚨 Detected {anomaly_count} anomalies that may affect analysis")
        
        return self.anomalies
    
    def predictive_analysis(self, target_column, test_size=0.2):
        """
        Predictive analysis using Random Forest
        """
        if target_column not in self.df.columns:
            print(f"[❌] Column {target_column} not found")
            return None
        
        # Prepare data
        numeric_df = self.df.select_dtypes(include=[np.number])
        if target_column not in numeric_df.columns:
            print(f"[❌] Column {target_column} must be numeric")
            return None
        
        X = numeric_df.drop(columns=[target_column, 'is_anomaly'], errors='ignore')
        y = numeric_df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"\n[🔮 Predictive Analysis - {target_column}]")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MSE: {mse:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n  Most important features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"    • {row['feature']}: {row['importance']:.4f}")
        
        self.predictions = {'model': model, 'r2': r2, 'mse': mse, 'features': feature_importance}
        
        if r2 > 0.7:
            self.insights.append(f"✅ Prediction model for {target_column} is very accurate (R²={r2:.2f})")
        elif r2 < 0.3:
            self.insights.append(f"⚠️ Prediction model for {target_column} needs improvement (R²={r2:.2f})")
        
        return self.predictions
    
    def correlation_intelligence(self, threshold=0.7):
        """
        Identify deep correlations between variables
        """
        corr_matrix = self.df.select_dtypes(include=[np.number]).corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_correlations.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        print(f"\n[🔗 Strong Correlations (|r| ≥ {threshold})]")
        for corr in strong_correlations:
            direction = "positive" if corr['correlation'] > 0 else "negative"
            print(f"  • {corr['var1']} ↔ {corr['var2']}: {corr['correlation']:.3f} ({direction})")
            self.insights.append(
                f"🔗 Strong {direction} correlation between {corr['var1']} and {corr['var2']} ({corr['correlation']:.2f})"
            )
        
        return strong_correlations
    
    def smart_recommendations(self):
        """
        Smart recommendations based on analysis
        """
        recommendations = []
        
        # Check variance
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            cv = self.df[col].std() / (self.df[col].mean() + 1e-10)
            if cv > 1.0:
                recommendations.append(f"📈 {col}: Needs normalization due to high variance")
        
        # Check anomalies
        if self.anomalies is not None and len(self.anomalies) > len(self.df) * 0.05:
            recommendations.append(f"🧹 Consider reviewing or removing anomalies ({len(self.anomalies)} values)")
        
        # Prediction recommendations
        if self.predictions and self.predictions['r2'] < 0.5:
            recommendations.append(f"🔧 Prediction model needs more variables or feature engineering")
        
        print(f"\n[💡 Smart Recommendations]")
        for rec in recommendations:
            print(f"  {rec}")
        
        return recommendations
    
    def correlation_heatmap(self):
        """Plot correlation heatmap"""
        plt.figure(figsize=(12,8))
        corr = self.df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
        plt.title("Correlation Heatmap", fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()
    
    def cluster_data(self, n_clusters=3):
        """Cluster data using KMeans"""
        numeric_data = self.df.select_dtypes(include=[np.number]).drop(columns=['is_anomaly'], errors='ignore')
        
        if numeric_data.shape[1] < 2:
            print("[⚠️] Not enough columns for clustering")
            return
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_data)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(reduced_data)
        self.df['Cluster'] = clusters
        
        print(f"\n[🎯 Clustering]")
        print(f"  Created {n_clusters} clusters")
        
        # Cluster statistics
        for i in range(n_clusters):
            count = (clusters == i).sum()
            print(f"  • Cluster {i}: {count} items ({count/len(clusters)*100:.1f}%)")
        
        plt.figure(figsize=(10,6))
        scatter = plt.scatter(reduced_data[:,0], reduced_data[:,1], c=clusters, cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Cluster')
        plt.title("Cluster Visualization (PCA + KMeans)", fontsize=14)
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def generate_ai_insights(self):
        """
        Generate intelligent report in natural language
        """
        print("\n" + "="*60)
        print("🤖 AI Intelligence Report")
        print("="*60)
        
        print(f"\n📋 Data Summary:")
        print(f"  • Total rows: {len(self.df):,}")
        print(f"  • Total columns: {len(self.df.columns)}")
        
        if self.insights:
            print(f"\n🔍 Discovered Insights:")
            for insight in self.insights:
                print(f"  {insight}")
        
        print("\n" + "="*60)
    
    def save_report(self, output_path="enhanced_report.csv"):
        """Save final report"""
        self.df.to_csv(output_path, index=False)
        print(f"\n[💾] Report saved to: {output_path}")


# ======== Usage Example ========
if __name__ == "__main__":
    # Load data
    analyzer = SmartDataAnalyzer("data.csv")
    
    # Clean data
    analyzer.clean_data()
    
    # Basic statistics
    analyzer.basic_statistics()
    
    # Detect anomalies
    analyzer.detect_anomalies(contamination=0.05)
    
    # Predictive analysis (example: predicting 'Sales' column)
    # analyzer.predictive_analysis('Sales')
    
    # Deep correlations
    analyzer.correlation_intelligence(threshold=0.6)
    
    # Smart recommendations
    analyzer.smart_recommendations()
    
    # Visualizations
    analyzer.correlation_heatmap()
    analyzer.cluster_data(n_clusters=4)
    
    # AI insights report
    analyzer.generate_ai_insights()
    
    # Save report
    analyzer.save_report("smart_analysis_report.csv")