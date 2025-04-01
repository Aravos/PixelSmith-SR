from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dir = "./02-Upscale-Project/Image-Upscaler/src/ProGAN/logs"

# Create an EventAccumulator
event_acc = EventAccumulator(log_dir)
event_acc.Reload()  # Load all events

# Get scalar tags (your loss values)
tags = event_acc.Tags()["scalars"]

# Print only loss-related metrics (case-sensitive check)
for tag in tags:
    if "Loss" in tag:  # Assuming your tags contain "Loss" (like "Loss Critic")
        events = event_acc.Scalars(tag)
        for event in events:
            print(f"Step {event.step}: {tag} = {event.value}")