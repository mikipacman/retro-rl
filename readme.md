# Retro RL Tournament
This repo contains tools that I built on top of OpenAI's [Retro](https://github.com/openai/retro)
gym in order to run a tournament between RL agents trained in retro games environments. For now
I'm focusing on fighting games like Street Fighter and Mortal Kombat. I'm planning on testing
selfplay techniques on these envs.

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