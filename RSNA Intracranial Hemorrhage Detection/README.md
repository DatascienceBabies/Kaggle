# RSNA Intracranial Hemorrhage Detection

[RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data)

## Setup
The training data is big (150GB+) and thus not contained in this repository.
Put this into the *data* folder which is excluded from source control. Adjust your scripts so they work against that folder.

Use the kaggle page to download it or use the kaggle api:

    pip install kaggle
	cd data
	kaggle competitions download -c rsna-intracranial-hemorrhage-detection
	
You will have to export an API token from your account settings before you can use the api. Check the [kaggle-api-github](https://github.com/Kaggle/kaggle-api) for more info.

