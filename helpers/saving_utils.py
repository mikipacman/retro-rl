from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
import neptune
import dill as pickle
import tempfile
import os
from datetime import datetime

params_pickle_name = "params.pkl"


# Needs to be used inside 'with neptune.create_experiment()'
def save_exp_params(params):
    with tempfile.TemporaryDirectory(dir="/tmp") as temp:
        path_to_pickle = os.path.join(temp, params_pickle_name)
        pickle.dump(params, open(path_to_pickle, "wb"))
        neptune.send_artifact(path_to_pickle, params_pickle_name)


def get_exp_params(exp_name, project_name):
    project = neptune.init(project_name)
    exp_list = project.get_experiments(exp_name)
    if len(exp_list) == 1:
        with tempfile.TemporaryDirectory(dir="/tmp") as temp:
            exp_list[0].download_artifact(params_pickle_name, temp)
            return pickle.load(open(os.path.join(temp, params_pickle_name), "rb"))
    else:
        raise Exception("Wrong exp id!")


class GoogleDriveCheckpointer():
    def __init__(self, project_experiments_path, exp_id):
        self._drive = GoogleDrive(GoogleAuth())
        self._project_experiments_path = project_experiments_path
        self._exp_id = exp_id

    def get_list_of_checkpoints(self):
        path_to_experiment = os.path.join(self._project_experiments_path, self._exp_id)
        folder = self._get_folder(path_to_experiment)
        children = self._get_children_list(folder["id"])
        children = sorted(children, key=lambda x: datetime.strptime(x["createdDate"], "%Y-%m-%dT%H:%M:%S.%fZ"))
        return [x["title"] for x in children]   # '2020-09-13T10:01:03.998Z'

    def download_checkpoints(self, checkpoints, destination_path):
        path_to_experiment = os.path.join(self._project_experiments_path, self._exp_id)
        folder = self._get_folder(path_to_experiment)
        children = self._get_children_list(folder["id"])
        children_to_download = [c for c in children if c["title"] in checkpoints]
        for c in children_to_download:
            c.GetContentFile(os.path.join(destination_path, c["title"]))

    def upload_checkpoint(self, name, model):
        path_to_experiment = os.path.join(self._project_experiments_path, self._exp_id)
        folder = self._get_folder(path_to_experiment, create=True)
        with tempfile.TemporaryDirectory(dir="/tmp") as temp:
            path = os.path.join(temp, name)
            model.save(path)

            file = self._drive.CreateFile({'title': name, 'parents': [{"id": folder["id"]}]})
            file.SetContentFile(path)
            file.Upload()

    def _add_folder(self, name, parent):
        meta = {'title': name, 'mimeType': 'application/vnd.google-apps.folder'}
        if parent:
            meta.update({"parents": [{"id": parent}]})
        folder = self._drive.CreateFile(meta)
        folder.Upload()
        return folder

    def _get_children_list(self, parent):
        q = "trashed=false"
        if parent:
            q += f" and '{parent}' in parents"
        return self._drive.ListFile({'q': q}).GetList()

    def _get_folder(self, path, create=False):
        parent = {}
        for sub_path in path.strip("/").split("/"):
            parent = self._get_folder_one_level(sub_path, parent.get("id"), create)

        return parent

    def _get_folder_one_level(self, title, parent, create=False):
        folder_list = self._get_children_list(parent)

        if title in [f["title"] for f in folder_list]:
            return [f for f in folder_list if f["title"] == title][0]
        elif create:
            return self._add_folder(title, parent)
        else:
            raise Exception("Folder does not exist!")
