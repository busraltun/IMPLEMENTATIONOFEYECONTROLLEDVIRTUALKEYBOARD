

# Create a custom function that generates a correctly formatted url
def choose_moustache(path=0):
    """This function violates every principle of a good function BUT your mates don't have time, they need to grow their mo NOW!"""
    import webbrowser
    import pandas as pd

    world_beard_championship = pd.read_csv("https://raw.githubusercontent.com/beccarobins/movember/master/world_beard_championship.csv", 
                                           index_col = 0)
    stache_df = world_beard_championship[world_beard_championship.index.str.contains("Moustache")==True].reset_index()
    stache_df.rename(columns={'index':'category'}, inplace=True)
    
    # Choose your mo
    for i, moustache_type in stache_df.iterrows():
        print(i, moustache_type.category)
    path = int(input('\nChoose the index of your preferred moustache: '))
    
    # Clean up paths and strings
    url = "https://www.austinfacialhairclub.com/2017-wbmc-results"
    new_url = url.replace("2017-wbmc-results", "")
    test_String = stache_df.iloc[path]['category'].replace(" ", "-")
    test = new_url+'results-'+test_String
    site = test.lower()
    
    # Use webbrowser module when packaging up to send to friends
    # Use IPython in Jupypter
    webbrowser.open(site)
    return site