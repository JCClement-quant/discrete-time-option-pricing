import streamlit as st

# Pages
page_1 = st.Page("pages/Tree_Explorer.py", title="Tree Explorer", icon="🌳", default=True)
page_2 = st.Page("pages/Cpp_Pricing_and_Convergence.py",
                 title="Binomial Pricing and Convergence", icon="🧮")

# Browsing
pg = st.navigation([page_1, page_2])

# Execution
pg.run()