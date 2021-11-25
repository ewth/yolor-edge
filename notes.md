# Notes

Scratchpad.

---

## TODO

### Now

- Get camera recognised
- Figure out how to run detection on camera
- Figure out how to stream result to host

### Later

- Get wandb working again.
- Check out/install mish-cuda
- Check out/install pytorch_wavelets

## 27/11/21

Focusing today on getting the model running with video.

First though, want to throw a list of how to get it going together. In the few days between working on it I've almost forgotten!

### Troubleshooting
The script `jetson-yolor/scripts/test.sh` wasn't working at first, I'm certain it was last time.
I had changed it to call `python3 /yolor/test.py` which may have upset it.
Running `cd /yolor` and then `python3 test.py` (like it was before) seemed to do the job.
However I also didn't pass any wandb args this time; checking that's not it.
Enabling wandb again, it crashed. Interesting. I would like to have wandb running, but it's not a huge priority for today.

Also just remembered, not supposed to invoke `scripts/test.sh` directly; supposed to run `scripts/yolor-p6.sh` (or other model name).

#### Wandb

Ran `pip3 install wandb --force-reinstall`
Still crashing.
Rebuilt the image.
Tried numerous other things.
Removed wanbd and it works.
So skipping over this for now, will return to wandb later.