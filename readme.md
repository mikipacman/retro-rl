# Retro RL Tournament
This repo contains tools that I built on top of OpenAI's [Retro](https://github.com/openai/retro)
gym in order to run a tournament between RL agents trained in retro games environments. For now
I'm focusing on fighting games like Street Fighter and Mortal Kombat. I'm planning on testing
selfplay techniques on these envs.

# Some cool results
I played with this repo at the begining of my journey with machine learning, during my first fascination with reinforcement learning. At that time I did not have any reasonable compute so I ran all the experiments on my laptop :D Even if the results were not SOTA it was really fun seeing my agent winning against a computer. 

Here is a particularly fun run (because succesful). The agent is playing `Mortal Kombat II` on `VeryEasy` mode (other cases were to much for my compute). We can see by the reward curve that the agent is in fact learning somehing!

![image](https://github.com/mikipacman/retro-rl/assets/33525262/bff59445-ed8f-4e97-8cd3-68d680f02453)

Now the best part, gameplays! First one is from the epoch number 0, so the agent fails obviously (we are playing `Scorpion`)


https://github.com/mikipacman/retro-rl/assets/33525262/50de6e69-9aed-4b95-836b-a2d7c1c28674


And in the epoch number 100  `SCORPION WINS!` (Watch with audio, I spent a lot of time on properly rendering these videos)

https://github.com/mikipacman/retro-rl/assets/33525262/be3da388-55eb-49d4-bbc3-a82ecefd35df




# Before you pip install
```
sudo apt-get update

sudo apt-get install git python3-dev python3-numpy libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libsdl1.2-dev libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev 

sudo apt-get install python3-mpi4py

sudo apt-get install swig

sudo apt-get install libportaudio2
```


# How to run those scripts?
- create empty conda env
- `pip install -r requirements.txt`
- download states (for now only [MortalKombat2](https://drive.google.com/file/d/1unUllgKxj1VInR-WxDxxQHZnoHsg1uDr/view?usp=sharing) is available)
- paste `MortalKombat2/ready_to_play/*` to `<path to python retro-rl in your venv>/data/stable/MortalKombatII-Genesis/`
- enjoy life with `python MortalKombat2/enjoy.py`
- possibly mess around with `train.py` and `play.py`
- choose your destiny
- train some powerful agents and rule the world!


# How to create own states?
Following instructions work for Sega Genesis emulator
- install [Retro](https://github.com/openai/retro)
- install [RetroArch](https://www.retroarch.com/)
- copy `<path to python retro-rl>/cores/genesis_plus_gx_libretro.so` to `<path to retro arch>/cores`
- run RetroArch
- load copied core
- run rom
- save state (press `f2` by default)
- copy saved state from `<path to retro arch>/states/Genesis Plus GX/` to `<path to python retro-rl>/retro/data/stable/<your game>/`
- gzip the copied state and name it as `your_state.state`
- that's it!

You can access your state in env like this:
```
env = retro.make("MortalKombat3", state="your_state")
```
