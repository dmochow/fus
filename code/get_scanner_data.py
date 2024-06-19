#!/usr/bin/env python
# coding: utf-8

# In[3]:


## Placeholder for a script that will download data from flywheel and save it to a local directory
import os
import flywheel
import flywheel.api.acquisitions_api
import flywheel.finder
#import flywheel.flywheel
#from pathlib import Path
#import logging
from getpass import getpass
import datetime
from time import sleep
import platform
import zipfile
import re
from crontab import CronTab


# In[4]:


#Specify the root project path here
root_data = "C:\\Users\\Charm\\projects\\test_folder"
reserved_characters = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
#Create an instance of AcquisitionsApi
AcquisitionsApi = flywheel.api.acquisitions_api.AcquisitionsApi()
#Enable long path names for Windows with Powershell Command.
#if os.system == 'Windows':
    #os.system('New-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem" \' -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force')
    #os.system('Set-ItemProperty -Path HKLM:SYSTEM\CurrentControlSet\Control\FileSystem -Name LongPathsEnabled -Value 1')
#else:
    #pass
    
    


# In[16]:


jobScheduler = CronTab(user = 'Charm')
job = jobScheduler.new(command = 'python get_scanner_data.ipynb')
job.minute.every(10)
jobScheduler.write()


# In[5]:


#Have the user enter their API key
api_key = getpass('Please enter your API key: ')
#Initialize the client 
fw = flywheel.Client(api_key)
#Delete the API key
del api_key


# In[6]:


#Create an instance of Finder
Finder = flywheel.finder.Finder(fw, 'iter_find')


# In[7]:


os.path.exists(root_data)
test_resolver_path = 'fw://group_id/project_id/subject_id/session_id/acquisition_id/files/file_id'
print(platform.system())


# In[10]:


datetime_now = '2024-06-10T00:00:00.000'
#Other parameters include: container.timestamp, container.modified, container.created, file.created, file.modified, file.size, file.name, file.type and file.classification.Measurement
#CASE SENSITIVE
group_label = input('Please type in the name of the group you would like to automatically download updates from: ')
project_label = input('Please type in the name of the project you would like to automatically download updates from: ')
#Temporarily, the project that we will iterate through is fus
nifti_query = f'file.created >= {datetime_now}Z AND project.label = {project_label} AND group.label = {group_label} AND file.type = nifti'
#json_query = f'file.created >= {datetime_now}Z AND project.label = {project_label} AND group.label = {group_label} AND file.type = json'
#dicom_query = f'file.created >= {datetime_now}Z AND project.label = {project_label} AND group.label = {group_label} AND file.type = dicom'

search_input = {'structured_query': nifti_query, 'return_type': 'file'}

new_files_list = fw.search(search_input, size = 10000)
print(new_files_list[0])
subject_id = new_files_list[0]['subject']['id']
file_name = new_files_list[0]['file']['name']
print(fw.files.find(f'name={file_name}'))
#18:33:14.582000+00:00
#print(new_files_list[0], new_files_list[0]['subject']['created'])
#subject = fw.get_subject(new_files_list[0]['subject']['id'])

    
#Finder objects: [0]['parents']['acquisition'] vs. Search objects: ['acquisition']
#*Note to Self: id = jacek label = JACEK
#Cannot index the id from the search object

#ReadTimeoutError: HTTPSConnectionPool(host='api.flywheel.io', port=443): Read timed out. (read timeout=30) => Likely occurring due to the sheer number of results 
        


# In[13]:


datetime_now = '2024-06-15T00:00:00.000'

#Returns a list of dictionaries.
file_ids = []
path = []
#Returns a list of dictionaries, will get index out of range error if the list is empty.
#new_acquisitions_list = fw.acquisitions.find('created>2024-05-15')
#List all possible file types.
file_types = ['nifti', 'json', 'dicom', 'zip']


# In[19]:


#Return datetime in the format year-month-dayThour:minute:second:millisecond
datetime_now = datetime.datetime.now() 
pattern = re.compile(r'\s')

print(datetime_now, pattern)

#Replace the space with a T
datetime_now = pattern.sub('T', str(datetime))

print(datetime_now)

datetime_now = str(datetime.datetime.now())
datetime_now = datetime_now.replace(' ', 'T')
print(datetime_now)


# In[32]:


*/10*2-6***1-6

def check_for_updates():
    global datetime_now 

    #datetime_now = str(datetime_now)
    #datetime_now = datetime_now.replace(' ', 'T')

    #User inputs the group and project they would like to check for updates from, might need to put these statements outside of the function if scheduling the function to run at a specific time.
    group_label = input('Please type in the name of the group you would like to automatically download updates from: ')
    project_label = input('Please type in the name of the project you would like to automatically download updates from: ')
    
    #Check for updates for all file types separately
    nifti_query = f'file.created >= {datetime_now}Z AND project.label = {project_label} AND group.label = {group_label} AND file.type = nifti'
    json_query = f'file.created >= {datetime_now}Z AND project.label = {project_label} AND group.label = {group_label} AND file.type = json'
    dicom_query = f'file.created >= {datetime_now}Z AND project.label = {project_label} AND group.label = {group_label} AND file.type = dicom'
    zip_query = f'file.created >= {datetime_now}Z AND project.label = {project_label} AND group.label = {group_label} AND file.type = zip'
    
    #Create a list of the search objects for the separate files
    query_list = [nifti_query, json_query, dicom_query, zip_query]

    for query in query_list:
        search_input = {'structured_query': query, 'return_type': 'file'}
        #Removed the size parameter to see if the error persists.
        new_files_list =fw.search(search_input, size = 10000)
        print(len(new_files_list))
        if len(new_files_list) == 0:
            pass
        else:
            #For all the search objects, get the containers names except for the acquisition, and use the .find function to get the 
            for new_file in new_files_list:
                print(new_files_list.index(new_file))
                file_name = new_file['file']['name']
                new_file_name = file_name
                subject_label = new_file['subject']['code']
                session_label = new_file['session']['label']
                find_file_object = fw.files.find(f'name={file_name}')
                #print(new_file_object)
                acquisition_id = find_file_object[0]['parents']['acquisition']
                acquisition = fw.get_acquisition(acquisition_id)
                
                for character in file_name:
                    if character in reserved_characters:
                        new_file_name = new_file_name.replace(character, '_')
                #print(type(new_file_name), new_file_name)
                #Create a new directory for the group in the local directory.
                try:
                    if platform.system() == 'Windows':
                        path = f'{root_data}\\{group_label}\\{project_label}\\{subject_label}\\{session_label}\\{acquisition.label}'
                        os.makedirs(path)
                    else:
                        path = f'{root_data}/{group_label}/{project_label}/{subject_label}/{session_label}/{acquisition.label}'
                        os.makedirs(path)
                except OSError:
                    pass
                if new_file not in os.walk(path):
                    if platform.system() == 'Windows':
                        acquisition.download_file(file_name, path + '\\' + new_file_name)
                    else:
                        acquisition.download_file(file_name, path + '/' + new_file_name)
    
    #Return datetime in the format year-month-dayThour:minute:second:millisecond
    datetime_now = datetime.datetime.now() 
    datetime_now = str(datetime.datetime.now())
    #Replace the space with a T
    datetime_now = datetime_now.replace(' ', 'T')
    return datetime_now
#ConnectionError
check_for_updates()
#sleep(60)


# In[ ]:


#Function downloads all data from specified group(s) to the local directory group by group (up down approach).
#*Note to Self: .lookup takes in the labels as input, .get takes in the ID as input.
def download_all_flywheel_data():
    if platform.system() != 'Windows':
        group_list = []
        project_list = []
        #For each group in the flywheel instance, check if the user would like to download all the data from the group, if yes, add the group to the group list.
        for group in fw.groups():
            user_group = input(f'Would you like to download all the data from the group {group.label}? (yes/no): ')
            if user_group == 'yes':
                group_list.append(group)
            elif user_group == 'no':
                continue
        for group in group_list:
            new_group_label = group.label
            for character in group.label: 
                if character in reserved_characters: 
                    new_group_label = new_group_label.replace(character, '_') 
            #Default value of exist_ok is False and will raise an OSError if the directory exists when using os.makedirs. If using Path.mkdir, the default value of exist_ok is False and will raise a FileExistsError if the directory exists. Path.mkdir takes in path, os.makedirs takes in strings.
            #Create a directory for the group in the local directory.
            try:
                os.makedirs(root_data + '/' + new_group_label)
            except OSError:
                pass
            #For each project in the group, check if the user would like to download all the data from the project, if yes, add the project to the project list.
            for project in group.projects():
                user_project = input(f'Would you like to download all the data from the project {project.label}? (yes/no): ')
                if user_project == 'yes':
                    project_list.append(project)
                elif user_project == 'no':
                    continue
            #Create a directory for the project in the local directory.
            for project in project_list:
                new_project_label = project.label
                for character in project.label: 
                    if character in reserved_characters: 
                        new_project_label = new_project_label.replace(character, '_')
                try:
                    os.makedirs(root_data + '/' + new_group_label + '/' + new_project_label)
                except OSError:
                    pass
                #Create a directory for the subject in the local directory.
                for subject in project.subjects():
                    new_subject_label = subject.label
                    for character in subject.label: 
                        if character in reserved_characters: 
                            new_subject_label = new_subject_label.replace(character, '_')
                    try:
                        os.makedirs(root_data + '/' + new_group_label + '/' + new_project_label + '/' + new_subject_label)
                    except OSError:
                        pass
                    #Create a directory for the session in the local directory.
                    for session in subject.sessions():
                        new_session_label = session.label
                        for character in session.label: 
                            if character in reserved_characters: 
                                new_session_label = new_session_label.replace(character, '_')
                        try:
                            os.makedirs(root_data + '/' + new_group_label + '/' + new_project_label + '/' + new_subject_label + '/' + new_session_label)
                        except OSError:
                            pass
                        for acquisition in session.acquisitions():
                            new_acquisition_label = acquisition.label
                            for character in acquisition.label:
                                if character in reserved_characters:
                                    new_acquisition_label = new_acquisition_label.replace(character, '_')
                            try:
                                os.makedirs(root_data + '/' + new_group_label + '/' + new_project_label + '/' + new_subject_label + '/' + new_session_label + '/' + new_acquisition_label)
                            except OSError:
                                pass
                            #For each file in the acquisition, check if the file exists in the local directory, if it doesn't, download it.
                            for file in acquisition.files:
                                new_file_name = file.name
                                acquisition = fw.get_acquisition(acquisition.id)
                                for character in file.name:
                                    if character in reserved_characters:
                                        new_file_name = new_file_name.replace(character, '_')
                                        
                                path = root_data + '/' + new_group_label + '/' + new_project_label + '/' + new_subject_label + '/' + new_session_label + '/' + new_acquisition_label
                                
                                if file not in os.walk(path):  
                                    #Download the file from the acquisition using the original file name but naming the path with the new file name.
                                    acquisition.download_file(file.name, path + '/' + new_file_name)
                                    #AcquisitionsApi.download_file_from_acquisition(acquisition_id = acquisition.id, file_name = file.name, dest_file = f'{path}/{file.name}')
                                        #file.download(f'{path}/{file.name}') 
                                else:
                                    #Create a duplicate file with a different name.
                                    print(f'There is a file with the name {file.name} already in the local directory. No action.')
    else:
        download_all_flywheel_data_windows()
    
#download_all_flywheel_data()


# In[33]:


def download_all_flywheel_data_windows():
    if platform.system() == 'Windows':
        group_list = []
        project_list = []
        #For each group in the flywheel instance, check if the user would like to download all the data from the group, if yes, add the group to the group list.
        for group in fw.groups():
            user_group = input(f'Would you like to download all the data from the group {group.label}? (yes/no): ')
            if user_group == 'yes':
                group_list.append(group)
            elif user_group == 'no':
                continue
        for group in group_list:
            new_group_label = group.label
            for character in group.label: 
                if character in reserved_characters: 
                    new_group_label = new_group_label.replace(character, '_') 
            #Default value of exist_ok is False and will raise an OSError if the directory exists when using os.makedirs. If using Path.mkdir, the default value of exist_ok is False and will raise a FileExistsError if the directory exists. Path.mkdir takes in path, os.makedirs takes in strings.
            #Create a directory for the group in the local directory.
            try:
                os.makedirs(root_data + '\\' + new_group_label)
            except OSError:
                pass
            #For each project in the group, check if the user would like to download all the data from the project, if yes, add the project to the project list.
            for project in group.projects():
                user_project = input(f'Would you like to download all the data from the project {project.label}? (yes/no): ')
                if user_project == 'yes':
                    project_list.append(project)
                elif user_project == 'no':
                    continue
            #Create a directory for the project in the local directory.
            for project in project_list:
                new_project_label = project.label
                for character in project.label: 
                    if character in reserved_characters: 
                        new_project_label = new_project_label.replace(character, '_')
                try:
                    os.makedirs(root_data + '\\' + new_group_label + '\\' + new_project_label)
                except OSError:
                    pass
                #Create a directory for the subject in the local directory.
                for subject in project.subjects():
                    new_subject_label = subject.label
                    for character in subject.label: 
                        if character in reserved_characters: 
                            new_subject_label = new_subject_label.replace(character, '_')
                    try:
                        os.makedirs(root_data + '\\' + new_group_label + '\\' + new_project_label + '\\' + new_subject_label)
                    except OSError:
                        pass
                    #Create a directory for the session in the local directory.
                    for session in subject.sessions():        
                        new_session_label = session.label
                        for character in session.label: 
                            if character in reserved_characters: 
                                new_session_label = new_session_label.replace(character, '_')
                        try:
                            os.makedirs(root_data + '\\' + new_group_label + '\\' + new_project_label + '\\' + new_subject_label + '\\' + new_session_label)
                        except OSError:
                            pass
                        for acquisition in session.acquisitions():
                            new_acquisition_label = acquisition.label
                            for character in acquisition.label:
                                if character in reserved_characters:
                                    new_acquisition_label = new_acquisition_label.replace(character, '_')
                            try:
                                os.makedirs(root_data + '\\' + new_group_label + '\\' + new_project_label + '\\' + new_subject_label + '\\' + new_session_label + '\\' + new_acquisition_label)
                            except OSError:
                                pass
                            #For each file in the acquisition, check if the file exists in the local directory, if it doesn't, download it.
                            for file in acquisition.files:
                                new_file_name = file.name
                                acquisition = fw.get_acquisition(acquisition.id)
                                for character in file.name:
                                    if character in reserved_characters:
                                        new_file_name = new_file_name.replace(character, '_')
                                path = root_data + '\\' + new_group_label + '\\' + new_project_label + '\\' + new_subject_label + '\\' + new_session_label + '\\' + new_acquisition_label
                                if file not in os.walk(path):
                                    #Download the file from the acquisition using the original file name but naming the path with the new file name.
                                    acquisition.download_file(file.name, path + '\\' + new_file_name)
                                else:
                                    #Create a duplicate file with a different name.
                                    print(f'There is a file with the name {file.name} already in the local directory. No action.')
    else:
        download_all_flywheel_data()
#download_all_flywheel_data_windows()


# In[14]:


def prepare_bids_windows():
    #anat -> sub, ses, task, acq, ce, rec, run, ... chunk, suffix (suffix, MEGRE, MESE, VFA, defacemask, IRT1, MP2RAGE, MPM, MTS) 
    #One anat path echo -> path, optional: flip -> inv or flip -> mt
    if platform.system() == 'Windows':
        project_list = []
        #For each group in the flywheel instance, check if the user would like to download all the data from the group, if yes, add the group to the group list.
            #For each project in the group, check if the user would like to download all the data from the project, if yes, add the project to the project list.
        group_id = input('Please type in the ID of the group you would like to download data from: ')
        group = fw.get_group(f'{group_id}')
        for project in group.projects():
            user_project = input(f'Would you like to download all the data from the project {project.label}? (yes/no): ')
            if user_project == 'yes':
                project_list.append(project)
            elif user_project == 'no':
                continue
            #Create a directory for the project in the local directory.
        for project in project_list:
            new_project_label = project.label
            for character in project.label: 
                if character in reserved_characters: 
                    new_project_label = new_project_label.replace(character, '_')
            try:
                os.makedirs(root_data + '\\' + new_project_label)
            except OSError:
                pass
            #Create a directory for the subject in the local directory.
            #print(type(project.subjects()))
            for subject in project.subjects():
                project_subject_list = project.subjects()
                print(project.subject.index(subject))
                new_subject_label = f'sub-{project_subject_list.index(subject) + 1: 02d}'
                try:
                    os.makedirs(root_data + '\\' + new_project_label + '\\' + new_subject_label)
                except OSError:
                    pass
                #Create a directory for the session in the local directory.
                for session in subject.sessions():
                    new_session_label = session.label
                    for character in session.label: 
                        if character in reserved_characters: 
                            new_session_label = new_session_label.replace(character, '_')
                    try:
                        os.makedirs(root_data + '\\' + new_group_label + '\\' + new_project_label + '\\' + new_subject_label + '\\' + new_session_label)
                    except OSError:
                        pass
                    for acquisition in session.acquisitions():
                        new_acquisition_label = acquisition.label
                        for character in acquisition.label:
                            if character in reserved_characters:
                                new_acquisition_label = new_acquisition_label.replace(character, '_')
                        try:
                            os.makedirs(root_data + '/' + new_group_label + '/' + new_project_label + '/' + new_subject_label + '/' + new_session_label + '/' + new_acquisition_label)
                        except OSError:
                            pass
                        #For each file in the acquisition, check if the file exists in the local directory, if it doesn't, download it.
                        for file in acquisition.files:
                            new_file_name = file.name
                            acquisition = fw.get_acquisition(acquisition.id)
                            for character in file.name:
                                if character in reserved_characters:
                                    new_file_name = new_file_name.replace(character, '_')
                                        
                            path = root_data + '/' + new_group_label + '/' + new_project_label + '/' + new_subject_label + '/' + new_session_label + '/' + new_acquisition_label
                                
                            if file not in os.walk(path):  
                                #Download the file from the acquisition using the original file name but naming the path with the new file name.
                                acquisition.download_file(file.name, path + '/' + new_file_name)
                                #AcquisitionsApi.download_file_from_acquisition(acquisition_id = acquisition.id, file_name = file.name, dest_file = f'{path}/{file.name}')
                                    #file.download(f'{path}/{file.name}') 
                            else:
                                #Create a duplicate file with a different name.
                                print(f'There is a file with the name {file.name} already in the local directory. No action.')
    else:
        pass
    
#prepare_bids_windows()
    


# In[18]:


project_list = []
        #For each group in the flywheel instance, check if the user would like to download all the data from the group, if yes, add the group to the group list.
            #For each project in the group, check if the user would like to download all the data from the project, if yes, add the project to the project list.
group_id = input('Please type in the ID of the group you would like to download data from: ')
group = fw.get_group(f'{group_id}')
for project in group.projects():
    user_project = input(f'Would you like to download all the data from the project {project.label}? (yes/no): ')
    if user_project == 'yes':
        project_list.append(project)
    elif user_project == 'no':
        continue
#Create a directory for the project in the local directory.
for project in project_list:
    new_project_label = project.label
    for character in project.label: 
        if character in reserved_characters: 
            new_project_label = new_project_label.replace(character, '_')
        try:
            os.makedirs(root_data + '\\' + new_project_label)
        except OSError:
            pass
#Create a directory for the subject in the local directory.
#print(type(project.subjects()))
        for subject in project.subjects():
            project_subject_list = project.subjects()
            print(project.subject.index(subject))
            new_subject_label = f'sub-{project_subject_list.index(subject) + 1: 02d}'
            try:
                os.makedirs(root_data + '\\' + new_project_label + '\\' + new_subject_label)
            except OSError:
                pass


# In[ ]:


## The data will then be ready to be pre-processed

