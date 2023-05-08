# online-semantic-mapping

## Pre-requisites
- python 3.8.16
- Bosdyn SDK
- CLIP
- Segment Anything

### Installing Bosdyn SDK

- You can get more information about the Bosdyn SDK here:
    - https://dev.bostondynamics.com/docs/python/quickstart

- You can install the SDK like so:
```
python3 -m pip install --upgrade bosdyn-client bosdyn-mission bosdyn-choreography-client
```

### Setting up hostname for Bosdyn SDK

- Specify the hostname and password for the spot robot in your ~/.bashrc file like so:
```
export BOSDYN_CLIENT_USERNAME="user"
export BOSDYN_CLIENT_PASSWORD="[ask me for password]"
```

## Running the application
- The application can be run with the following:
```
./run_app.sh
```
