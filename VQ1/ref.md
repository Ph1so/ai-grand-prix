What is AI Grand Prix? 

The AI Grand Prix, founded by Anduril in partnership with the Drone Champions League (DCL), Neros Technologies, and JobsOhio, is a global autonomous drone racing competition challenging the world’s best engineers to prove their autonomy software under real-world flight conditions. Teams and individuals from around the globe are invited to develop AI systems that pilot high-speed racing drones through dynamic, professional-grade courses with zero human control. Individuals or teams of up to 8 will compete for a $500,000 prize pool and a job at Anduril. All drones are identical and supplied by Neros. More competition details to come.

---

When and where will it take place? 

The competition kicks off with a virtual qualification phase in April-June, where teams from around the globe will submit their custom Python-based AI algorithms into a DCL-built platform and take on a challenging virtual racecourse. Top-performing teams will then advance to an exclusive, two-week, in-person training and qualification experience in September 2026 in Southern California – bringing the competition from a simulation to the real world. Season 1 culminates with the AI Grand Prix in Ohio, where the world’s best will compete on the biggest stage for $500,000 and to win a job at Anduril.

---

Who can participate and how to register? 

Registration is open to individuals, university teams, research organizations and those with a passion for AI programming. No professional credentials or certifications are required.To get started, register through this page. You’ll be the first to receive updates on next steps.

---

Is there an age restriction to sign up? 

All ages are allowed, but parents/guardians must register for any interested parties below the age of 14. Written parental consent and age verification will be required for all minors.

---

Does AI Grand Prix cover flights, accommodations, or other expenses? 

No. Each participating team is responsible for covering its own expenses related to the AI Grand Prix, including travel, accommodations, and any additional costs incurred. The Top 10 performing teams at the AI Grand Prix Ohio will be guaranteed a cash prize of at least $5,000.

---

Which drones will be used? Can teams use their own drone? 

Competitors will race fully autonomous, identical drones built by Neros Technologies, incorporating DCL’s AI vector module. All drones will be provided by the AI Grand Prix. Technical specifications will be shared at a later stage.

---

What does “Win a Job at Anduril” mean? 

Winning a job implies one role at Anduril for the 2026 competition season. Standard work eligibility requirements apply. Winning participants can select an alternative cash bonus if they do not meet the necessary eligibility requirements, choose not to pursue a job at Anduril, or if another team member from the winning team is hired. Additionally, top university performers that reach the physical qualifier stage in Southern California will be screened in person for potential internship and full-time entry-level roles at Anduril.

---

Is this a one-time event? 

No. The AI Grand Prix is an annual competition, with future Grand Prix events taking place around the globe.

---

Are teams allowed to apply? 

Participants can compete as individuals or as teams of up to 8 people. For teams, the team leader should register at this stage. More information on additional team members to be collected post registration.

---

Is there a registration fee? 

No. There is no registration fee.

---

What happens after registration? 

After you register, all important information will be sent via email, including software download links, key deadlines, and updates as the competition progresses.

---

When will the simulator and interface specs be released? 

Interface specifications will be released in the second half of March. The simulator package (including the course) releases in May. Updates, parameters, and previews will also be shared via weekly newsletters and website FAQs.

---

What sensors will be available in the virtual and physical environments? 

In the virtual environment, you’ll have access to typical FPV drone sensors (e.g., accelerometer, gyroscope, and likely motor RPM readouts). In the physical qualifier, you’ll also likely receive additional information such as battery status.

---

Will there be access to coordinates or absolute positions? 

Teams may receive limited coordinate information for the starting position in the virtual qualifier. Beyond that, you should expect to fly without coordinate/position data.

---

What vision hardware is available (camera / LiDAR)? 

The drone will have a single FPV camera (approximately a 12MP wide-angle camera). LiDAR will not be available. The challenge is to find a clear path without reliable depth information.

---

Do sensor models or drone models vary in the virtual qualifiers? 

No. There is one standardized virtual drone for the virtual qualifiers. The interface remains consistent between both rounds; the environment becomes more complex in the second round.

---

How do Virtual Qualifier Round 1 and Round 2 differ? 

Round 1 is intentionally simple, with a small number of gates and a desaturated environment with visually highlighted gates designed to help teams get started. Round 2 is significantly more realistic and visually complex, including a real 3D-scanned environment and a more challenging course.

---

Will there be wind or external disturbances in the virtual qualifiers? 

No. There will be no wind in the virtual qualifiers. Difficulty increases primarily through realism and visual complexity.

---

Is qualifying head-to-head or time-trial? 

Qualifying is a time-trial. Everyone runs the same course, and the fastest valid times advances.

---

Will teams get the map/course ahead of time? 

Teams will be able to download the simulator package (including the course) once it is released. Teams can attempt runs any time within the qualification window; once all gates are passed, the run time is locked.

---

Will Python code run on the physical drone too? 

Yes. The intention is that your Python-based approach transfers from the virtual environment to the physical drone, although some adaptation should be expected.

---

Can flight data be exported from the simulator? 

Yes. Teams should have access to flight data logs for analysis and iteration.

---

How is scoring handled (time, missing gates, penalties)? 

Scoring is primarily time-based but runs must successfully pass gates to count. Teams may need to balance speed versus reliability (faster control can increase missed-gate risk).

---

Does the drone know the track/map? Is SLAM required? 

The drone will not “know” the track. Teams must detect gates and navigate using onboard sensing (primarily vision). Gate position details may be provided only at a rough level; flight-path optimization is the team’s responsibility.

---

What onboard compute will be allowed (CPU/GPU/TOPS)? 

Exact compute details will be shared later. At a high level, the drone will use an AI-focused compute module (more capable than a Raspberry Pi), with a rough indication around ~100 TOPS depending on configuration and power limits.

---

What libraries are allowed? Python only or also C/Cython? 

Teams will broadly be able to use the libraries they need, with potential restrictions communicated later. Python is the primary interface, and compiled extensions (e.g., C/Cython) are generally expected to be possible.

---

What is included in the AI compute module? 

At a high level, the module is expected to include onboard AI compute with RAM/SSD, an FPV camera feed, IMU sensors (gyros/accelerometers), and connectivity modules such as Wi-Fi/Bluetooth. Using connectivity to pilot the drone would result in disqualification (the flight must be autonomous).

---

Can teams have sponsors, and can sponsor logos be placed on the drone? 

Yes, more on this to be shared later.

---

What’s the rough timeline (virtual qualifier, physical qualifier, finals)? 

The virtual qualification phase runs May – July, with a cutoff for Round 2 targeted around the end of July (approx.). Selected teams will be invited to an in-person physical qualifier in September 2026 in Southern California. The final event – the AI Grand Prix Ohio – is planned for November. More competition details to come.

---

Is the physical qualifier indoor or outdoor? 


Indoor, with consistent lighting for fairness. Expect obstacles and visual “distractions.” The finals will also be indoors and may include additional disturbances due to spectators (e.g. cameras/flash).