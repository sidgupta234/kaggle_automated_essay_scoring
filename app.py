import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from tester import compute_vocab_conventions_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt
showWarningOnDirectExecution = False
#from billboard_seaborn import *
st.set_page_config(page_title='Essay Rater', page_icon = "favicon.png", layout = 'wide', initial_sidebar_state = 'auto')
footer="""<style>
a:link , a:visited{
color: white;
background-color: transparent;
text-decoration: underline;
}
a:hover,  a:active {
color: blue;
background-color: transparent;
text-decoration: underline;
}
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: green;
color: white;
text-align: center;
}
</style>
<div class="footer">
<p><b>Developed by <a style='text-align: left;' href="https://www.languageof.me/" target="_blank">Siddharth Gupta</a> for NLP/NLU enthusiasts</b>
</p>
</div>
"""
#st.set_page_config(layout="wide")
c1, c2 = st.columns((2, 1))
header = st.container()
dataset = st.container()
spotify_features = st.container()
st.markdown(footer, unsafe_allow_html=True)
#modelTraining = st.container()

with header and c1:
     st.title("How good are your Vocabulary and Conventions?")
     st.write("Writing is a foundational skill. Sadly, it's one few students are able to hone, often because writing tasks are infrequently assigned in school. A rapidly growing student population, students learning English as a second language, known as English Language Learners (ELLs), are especially affected by the lack of practice. While automated feedback tools make it easier for teachers to assign more writing tasks, they are not designed with ELLs in mind.")
     st.write("This project is aimed to build a submission for the Kaggle contest (https://www.kaggle.com/competitions/feedback-prize-english-language-learning)")

     st.write("Essays on this website will be rated from a scale of 1-5 on the basis of Vocabulary and Conventions.")
with c2:
    essay = st.text_input("Enter your Essay to get it rated", value="Hello World")
    df = pd.DataFrame([essay], columns=["full_text"])
    #print(df)
    values = compute_vocab_conventions_score(df)
    vocab = "Vocabulary score is " + str(max(values[0][0], 1))
    conventions = "Conventions score is " + str(max(values[1][0], 1))
    st.text(vocab)
    st.text(conventions)
    fig, ax = plt.subplots()
    wordcloud = WordCloud(stopwords=None, min_word_length=0).generate(essay)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot(fig)

with c1:
    wordcloud = WordCloud().generate(essay)

#with dataset and c1:

#     song_iframe = st.empty()
#     # st.header("Billboard Top 100 Lyrics dataset")
#     # st.text("Billboard dataset augmented with Lyrics using Genius API")
#     file_path = "data/lyrics_lemmatized.csv"
#     music_data = preprocess__(file_path)
#     # music_data = pd.read_csv(file_path,  index_col=None)
#     # #st.write(music_data.head())
#     #st.write(music_data.index)
#     # st.write(music_data.columns)
#     # #music_data["performer+song"] = music_data["performer"] + "^" + music_data["song"]
#     music_data = music_data.set_index('performer+song')

#    # st.write(music_data.head())
#     #st.write(music_data.index)
#     #st.write(music_data.columns)
#     #df = pd.read_csv('data.csv')  
#     #set_of_artists = ""
#     set_of_artists = set(music_data['performer'].tolist())
#     artist_selected = st.selectbox('Select Artist', set_of_artists)
#     #st.write((music_data[music_data["performer"].str.contains(artist_selected)]))
    
#     # if(set_of_artists != ""):
#     artist_rows_with_aliases = music_data[music_data['performer'].str.contains(artist_selected)]
#     #st.write(artist_rows_with_aliases["performer"])
#     select_songs_by_artist_aliases = set(artist_rows_with_aliases['song'].tolist())
#     song_selected = st.selectbox('Select Song by your Artist', select_songs_by_artist_aliases)
#     #st.write()
#    # st.write('|'.join(artist_rows_with_aliases["performer"]))
#     row_to_print = music_data[(music_data['performer'].str.contains('|'.join(artist_rows_with_aliases["performer"]))) & (music_data["song"] == song_selected)].copy() #["Lyrics"]#==song_selected])]
    
#     #### Spotify_Song_ID_Details 
#     spotify_song_id_details = ["songid", "spotify_genre", "spotify_track_id", 
#     "spotify_track_preview_url", "spotify_track_duration_ms", "spotify_track_explicit", "spotify_track_album"]
    
#     # play_song = st.button('Play the selected track')

#     # if(play_song):
#     #     spotify_play_song(row_to_print[spotify_song_id_details])#[spotify_song_id_details])
#     # ####

#     recommend_songs_button = st.button('Playlist of songs like this!')

#     # embed_link = "https://open.spotify.com/embed/playlist/2tzzmfQ9OL0r4Sc5hgLHtP"
#     # components.iframe("https://open.spotify.com/embed/album/1DFixLWuPkv3KT3TnV35m3")

#     #st.markdown('<html>hey</html>', unsafe_allow_html=True)

#     # components.iframe(album_uri_link , width=600, height=200 )
#     # st.markdown('<html><iframe src="" width="300" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media></iframe></html>', unsafe_allow_html=True)
#     try:
#         song_id = row_to_print['spotify_track_id'].iloc[0]
#         embed_track_link = "https://open.spotify.com/embed/track/"+song_id
#         #components.iframe(embed_track_link, width = 300, height = 380 , scrolling = True)
#         song_iframe.markdown("<iframe src=" +str(embed_track_link)+" width = 300 height = 240></iframe>", unsafe_allow_html=True)
#     except:
#         pass
    
#     if(recommend_songs_button):
#         list_of_recommended_songs = recommend_songs(row_to_print, music_data)
#         #st.text(list_of_recommended_songs[['song', "performer"]].to_string(index=False))
#         playlist_link = make_playlist(row_to_print.iloc[0]['song'], list_of_recommended_songs)

#         html_for_link = "<p><a href="+str(playlist_link)+" style=color:#CDCDCD>Click here to go to the Playlist!</a></p> <div id='gtx-trans' style='position: absolute; left: 115px; top: -2px;'> <div class='gtx-trans-icon'></div> </div>"

#         st.write((html_for_link), unsafe_allow_html=True)
#         playlist_id = playlist_link.split("/")[-1]
#         embed_playlist_link = "https://open.spotify.com/embed/playlist/"+playlist_id
#         #components.iframe(embed_playlist_link)#, width = 300, height = 380 , scrolling = True)
#         st.markdown("<iframe src=" +str(embed_playlist_link)+" width = 300 height = 240></iframe>", unsafe_allow_html=True)


#     #iloc_index = artist_selected + "^" + song_selected
#     #print(iloc_index)
#     # st.text(row_to_print.iloc[0]["weekid"])
#     # st.text(row_to_print["song"][0])
#     # st.text(row_to_print["performer"][0])
    
#     #### Billboard Visualization
#     billboard_features = ["weekid", "week_position", "song", "performer", "instance", "peak_position", "weeks_on_chart", "spotify_genre"]
#     # st.text(row_to_print[billboard_features])

# with dataset and c2:
#     st.header("Lyrics for " +row_to_print["song"][0] + " by " +  row_to_print["performer"][0])
#     #st.text("Lyrics are")
#     #st.text(row_to_print["spotify_track_preview_url"][0])
#     st.text(row_to_print["Lyrics"][0])
#     #st.text(music_data[(music_data["performer"]==artist_selected) & (music_data["song"]==song_selected)]["Lyrics"])# & music_data["song"]==song_selected] )
    
# with dataset and c3:
#     st.header("Billboard Ranking Vs Time")
#     plt = billboard_viz_plot_weekid_vs_rank(row_to_print[billboard_features])
#     st.pyplot(plt)
#     #st.line_chart(row_to_print[billboard_features].iloc[0](columns={'date':'index'}).set_index('index'))
#     # print(type(row_to_print))
#     # print(row_to_print.columns)
#     # row_to_print.set_index("weekid", inplace = True)
#     # st.line_chart(row_to_print['week_position'].iloc[0])#, index="weekid")
#     # ####

# with dataset and c3:
#     #### Spotify_Audio_Features_Visualised
#     st.header("Audio properties of song")
#     audio_features = ["speechiness", "acousticness", "liveness", "valence", "danceability", "energy"]
#     #st.write(row_to_print.iloc[0])
#     plt = viz_song_with_audio_features(row_to_print[audio_features])
#     st.pyplot(plt)
#     #####
# Footer
# © 2022 GitHub, Inc.
# Footer navigation
# Terms
# Privacy
# Security
# Status
# Docs
# Contact GitHub
# Pricing
# API
# Training
# Blog
# About
# billboard-data-viz-and-lyrics-/app.py at main · sidgupta234/billboard-data-viz-and-lyrics-