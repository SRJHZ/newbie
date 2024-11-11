#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import subprocess
import sys

def main():
    subprocess.run([sys.executable, "-m", "streamlit", "run", "Forecast_Model.py"])

if __name__ == "__main__":
    main()

