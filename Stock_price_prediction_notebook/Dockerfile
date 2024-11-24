# Use an official Jupyter base image
FROM jupyter/scipy-notebook

# Copy your Jupyter Notebook files into the container
COPY Stock_Price_Prediction_and_Trading_Strategy.ipynb /home/jovyan/work/

# Set the working directory
WORKDIR /home/jovyan/work/

# Expose the Jupyter Notebook port
EXPOSE 8888

# Command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
