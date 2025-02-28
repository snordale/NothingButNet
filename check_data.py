from nothingbutnet.data_collector import NBADataCollector; collector = NBADataCollector(); games_df, _ = collector.fetch_basketball_reference(); print(f"Total games fetched: {len(games_df)}")
