# %%
import webknossos as wk

with wk.webknossos_context(token="<insert token here>", url="https://webknossos.esrf.fr"):
    # Download the dataset.
    dataset = wk.Dataset.download(
        dataset_name_or_url="MW_1984_vnc_050nm",
        organization_id="ESRF",
        webknossos_url="https://webknossos.esrf.fr"
    )