import pandas as pd
import glob
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def phase(over):
    if over <= 6:
        return "Powerplay"
    elif over <= 15:
        return "Middle"
    else:
        return "Death"

def main():
    print("Starting data consolidation...")
    data_dir = r"C:\Users\rowdy\Downloads\ipl_csv2 (2)"
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    df_list = []
    
    # Step 1 & 2: Load all delivery files (ignore info files)
    for f in files:
        if "_info" not in f:
            try:
                temp = pd.read_csv(f)
                df_list.append(temp)
            except Exception as e:
                print(f"Error reading file {f}: {e}")
                
    if not df_list:
        print("No data files found to consolidate!")
        return

    print(f"Loaded {len(df_list)} match files. Concatenating...")
    df = pd.concat(df_list, ignore_index=True)
    
    print("Initial Data Shape:", df.shape)

    # Step 3: Data Cleaning
    print("Cleaning data...")
    # Wicket type might be null if no wicket fell. We fill with 0 or empty string depending on type, but for now fillna(0) as per tutorial
    df.fillna(0, inplace=True)
    
    # Ensure required columns exist, some datasets might have slightly different names, but Cricsheet standard usually is:
    # match_id, season, start_date, venue, innings, ball, batting_team, bowling_team, striker(batter), non_striker, bowler, runs_off_bat, extras, etc.
    # The prompt specified: ["match_id","over","ball","batter","bowler","runs_off_bat","wicket_type"]
    
    # Wait, the Cricsheet header for the batsman is often 'striker' instead of 'batter'. Let's rename if needed.
    if 'striker' in df.columns and 'batter' not in df.columns:
        df.rename(columns={'striker': 'batter'}, inplace=True)
    
    # Some cricsheet datasets encode 'ball' as 0.1, 0.2 where whole number is over. Let's check structure.
    # But let's stick to the prompt's provided list as much as we can, we might need to adjust based on actual columns.
    cols_to_keep = ["match_id", "batter", "bowler", "runs_off_bat", "wicket_type"]
    
    # If 'over' doesn't exist, but 'ball' is like 0.1, we extract over.
    # Let's keep all columns first, then create features, then filter just in case.
    
    # If over doesn't exist but ball exists as float:
    if 'over' not in df.columns and 'ball' in df.columns:
        # e.g. ball = 0.1 -> over = 0
        df['over'] = df['ball'].astype(int)
        df['ball_num'] = ((df['ball'] - df['over']) * 10).round().astype(int)
    
    # Create phase
    df["phase"] = df["over"].apply(phase)

    # Example 2 — Dot ball column
    df["dot_ball"] = df["runs_off_bat"].apply(lambda x: 1 if x == 0 else 0)

    # Example 3 — Boundary column
    df["boundary"] = df["runs_off_bat"].apply(lambda x: 1 if x in [4, 6] else 0)
    
    # Make sure we only keep relevant columns if they exist
    available_cols = [c for c in ["match_id", "over", "ball", "batter", "bowler", "runs_off_bat", "wicket_type", "phase", "dot_ball", "boundary"] if c in df.columns]
    df = df[available_cols]

    print("Data processed. Saving cleaned_ipl_data.csv...")
    df.to_csv("cleaned_ipl_data.csv", index=False)
    
    print("Training RandomForest model for Run Prediction (Pre-Delivery Context)...")
    # Step 7: Machine Learning (Run Predictor)
    if "runs_off_bat" in df.columns and "over" in df.columns and "batter" in df.columns and "bowler" in df.columns:
        # Encode categorical variables
        from sklearn.preprocessing import LabelEncoder
        le_batter = LabelEncoder()
        le_bowler = LabelEncoder()
        
        # Fit encoders on the whole dataset to ensure all names are covered
        df['batter_encoded'] = le_batter.fit_transform(df['batter'].astype(str))
        df['bowler_encoded'] = le_bowler.fit_transform(df['bowler'].astype(str))
        
        # Handle 'ball' feature 
        if 'ball' in df.columns:
            if df['ball'].dtype == float:
                df['ball_num'] = ((df['ball'] - df['ball'].astype(int)) * 10).round().astype(int)
            else:
                df['ball_num'] = df['ball']
        else:
            df['ball_num'] = 1
            
        features = ["over", "ball_num", "batter_encoded", "bowler_encoded"]
        
        X = df[features]
        y = df["runs_off_bat"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=50, max_depth=7, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print(f"Model trained. Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        # Save model and label encoders
        joblib.dump({
            "model": model,
            "le_batter": le_batter,
            "le_bowler": le_bowler
        }, "run_model.pkl")
        print("Model and encoders saved to run_model.pkl.")
    else:
        print("Required columns for ML not found. Skipping model training.")

    print("Data preparation complete.")

if __name__ == "__main__":
    main()
