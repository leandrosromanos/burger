# Αυτόνομη εξερεύνηση TurtleBot3 Burger — ROS 2 Jazzy

Αυτό το πακέτο προσθέτει επιλογή στόχων από τον ζωντανό χάρτη του Cartographer.
Το Nav2 σχεδιάζει τη διαδρομή, αποφεύγει εμπόδια και εκτελεί ανακτήσεις.
Ο εξερευνητής δεν δημοσιεύει στο `/cmd_vel`.

**Κλείσε το obstacle_avoidance_v2.py, το παλιό obstacle_avoidance και το keyboard
teleop πριν ξεκινήσεις.** Μην στέλνεις παράλληλα στόχους από το RViz.

Αφορά χαρτογράφηση των προσβάσιμων περιοχών, όχι διαδρομή που περνά πάνω από κάθε
σημείο του δαπέδου. Δεν μπορεί να εγγυηθεί χαρτογράφηση πίσω από κλειστές πόρτες,
σε μη προσβάσιμα περάσματα ή σε σημεία που δεν βλέπει ο αισθητήρας.

## Αρχεία

- `frontier_explorer.py`: επιλογή frontiers, στόχοι Nav2, επαναπροσπάθειες, κατάσταση.
- `prepare_nav2.py`: δημιουργεί νέο `nav2_exploration.yaml` από τις προεπιλογές
  της εγκατεστημένης έκδοσης Nav2. Δεν τροποποιεί τα εγκατεστημένα πακέτα.
- `test_frontier.py`: αυτοματοποιημένοι έλεγχοι χωρίς ROS.

## 1. Τοποθέτηση και προαπαιτούμενα

Αποσυμπίεσε το ZIP μέσα στο `~/Projects/burger`. Θα δημιουργηθεί ο φάκελος:

```text
~/Projects/burger/turtlebot3_frontier
```

Χρειάζονται η υπάρχουσα εγκατάσταση TurtleBot3/Gazebo/Cartographer και τα Nav2,
NumPy και PyYAML. Αν λείπουν τα επιπλέον πακέτα:

```bash
sudo apt update
sudo apt install ros-jazzy-navigation2 ros-jazzy-nav2-bringup python3-numpy python3-yaml
```

Σε ΚΑΘΕ νέο τερματικό εκτέλεσε πρώτα:

```bash
source /opt/ros/jazzy/setup.bash
source ~/turtlebot3_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger
```

## 2. Gazebo — τερματικό 1

Αν είναι ήδη ανοικτό, μην ξεκινήσεις δεύτερο αντίγραφο.

```bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

Εναλλακτικά χρησιμοποίησε `turtlebot3_house.launch.py`. Το `empty_world` δεν είναι
κατάλληλο για να ελέγξεις πότε ολοκληρώνεται η εξερεύνηση ενός κλειστού δωματίου.

## 3. Cartographer — τερματικό 2

Αν ήδη τρέχει το SLAM σου, κράτησέ το. Διαφορετικά:

```bash
ros2 launch turtlebot3_cartographer cartographer.launch.py use_sim_time:=True
```

Αναμένονται `/map`, `/scan`, `/odom` και αλυσίδα TF `map -> odom -> base_footprint`.
Το Cartographer παρέχει τον χάρτη και το `map -> odom`.

## 4. Ρυθμίσεις και Nav2 — τερματικό 3

```bash
cd ~/Projects/burger/turtlebot3_frontier
python3 prepare_nav2.py
ros2 launch nav2_bringup navigation_launch.py \
  use_sim_time:=true \
  autostart:=true \
  use_composition:=false \
  params_file:="$PWD/nav2_exploration.yaml"
```

Χρησιμοποιούμε **navigation_launch.py**, δηλαδή μόνο πλοήγηση. Δεν ξεκινάμε
παράλληλα AMCL, map_server με αποθηκευμένο χάρτη, ή δεύτερο SLAM. Το συνηθισμένο
launch για πλοήγηση σε αποθηκευμένο χάρτη είναι διαφορετικό σενάριο.

Οι ρυθμίσεις κρατούν χαμηλή ταχύτητα: έως 0.06 m/s εμπρός, 0.04 m/s πίσω και
0.4 rad/s περιστροφή. Ο global planner δεν επιτρέπεται να σχεδιάζει μέσα σε
άγνωστο χώρο. Το Nav2 χρησιμοποιεί κυκλικό περίγραμμα ακτίνας 0.15 m συν padding
0.01 m· ο εξερευνητής είναι πιο συντηρητικός, με clearance 0.18 m και επιπλέον
περιθώριο διακριτοποίησης του χάρτη. Έλεγξε το περίγραμμα στο RViz αν έχεις
προσθέσει εξαρτήματα στο Burger.

Προεπιλογή εντολών είναι **TwistStamped**, όπως στο προηγούμενο πρόγραμμά σου.
Αν η βάση σου περιμένει `geometry_msgs/msg/Twist`, σταμάτησε το Nav2, τρέξε
`python3 prepare_nav2.py --twist` και ξεκίνησε ξανά το Nav2 με την ίδια εντολή.
Μην κάνεις αυτή την αλλαγή χωρίς να ελέγξεις τον τύπο του subscriber της βάσης:

```bash
ros2 topic info /cmd_vel --verbose
```

## 5. Εκκίνηση εξερευνητή — τερματικό 4

Περίμενε να ενεργοποιηθεί το Nav2 και να εμφανιστεί χάρτης στο RViz:

```bash
cd ~/Projects/burger/turtlebot3_frontier
python3 frontier_explorer.py --ros-args -p use_sim_time:=true
```

Δεν χρειάζεται `colcon build` για αυτά τα αυτόνομα Python αρχεία.
Αν το δικό σου TF χρησιμοποιεί μόνο `base_link`:

```bash
python3 frontier_explorer.py --ros-args \
  -p use_sim_time:=true -p base_frame:=base_link
```

## Πώς αποφασίζει

1. Εντοπίζει γνωστά ελεύθερα κελιά δίπλα σε άγνωστα κελιά, τα frontiers.
2. Ομαδοποιεί τα συνδεδεμένα frontiers και ελέγχει συνδεσιμότητα με τη θέση του ρομπότ.
3. Αναζητά σημεία παρατήρησης σε γνωστό ελεύθερο χώρο, με απόσταση από τα εμπόδια
   και τον άγνωστο χώρο. Ελέγχει επίσης ορατότητα προς το frontier στον στατικό χάρτη.
4. Προτιμά μικρότερη εκτιμώμενη διαδρομή και μεγαλύτερα frontiers. Το Nav2 κάνει
   τον τελικό σχεδιασμό και την αποφυγή με τα ζωντανά δεδομένα αισθητήρων.
5. Περιμένει να τελειώσει ο προηγούμενος στόχος ή να επιβεβαιωθεί η ακύρωσή του
   πριν στείλει καινούριο. Δεν αλλάζει στόχο σε κάθε ενημέρωση χάρτη.
6. Ακυρώνει στόχο μετά από 240 s ή μετά από 40 s χωρίς μετατόπιση τουλάχιστον
   0.12 m. Η μετατόπιση είναι εκτίμηση από TF και μπορεί να επηρεάζεται από SLAM.
7. Αποφεύγει προσωρινά αποτυχημένα σημεία για 60 s και επιτυχημένα για 20 s.
   Μετά από 3 επισκέψεις στην ίδια μικρή περιοχή δεν την επιλέγει ξανά στην
   ίδια εκτέλεση, ώστε να μη γυρίζει ατέρμονα σε frontier που δεν εξαφανίζεται.

Τα κινούμενα εμπόδια δεν είναι εγγυημένα στον `/map`: γι' αυτά ευθύνονται τα
costmaps και ο τοπικός ελεγκτής του Nav2. Η λογική παραμένει συντηρητική και
μπορεί να μην επιλέξει ένα πολύ στενό πέρασμα ακόμη κι αν χωρά οριακά το ρομπότ.

## Μηνύματα και ολοκλήρωση

| Μήνυμα | Τι σημαίνει |
|---|---|
| `WAIT_MAP` | Δεν υπάρχει πρόσφατος έγκυρος χάρτης. |
| `WAIT_TF` | Δεν υπάρχει πρόσφατη θέση του ρομπότ στον χάρτη. |
| `WAIT_NAV2` | Δεν είναι διαθέσιμο το action `/navigate_to_pose`. |
| `NAVIGATING` | Στάλθηκε ένας στόχος στο Nav2. |
| `REACHED` | Ο στόχος ολοκληρώθηκε. Περιμένει λίγο για ενημέρωση του χάρτη. |
| `CANCELING` | Ζητήθηκε ακύρωση· δεν στέλνει νέο στόχο μέχρι να τελειώσει ο παλιός. |
| `UNRESOLVED` | Υπάρχουν frontiers, αλλά οι θέσεις είναι μη ασφαλείς, σε αναμονή ή έχουν εξαντλήσει τις επισκέψεις. **Δεν σημαίνει ολοκλήρωση.** |
| `BLOCKED` | Η θέση του ρομπότ δεν βρίσκεται σε ελεύθερο χώρο με το απαιτούμενο περιθώριο. Έλεγξε χάρτη/TF/περίγραμμα. |
| `COMPLETE_REACHABLE` | Για τουλάχιστον 15 s και 3 διαφορετικές λήψεις χάρτη δεν υπάρχουν προσβάσιμα frontiers με τα τρέχοντα περιθώρια. |

Το `COMPLETE_REACHABLE` δεν δηλώνει «100% του φυσικού δωματίου». Αναφέρει επίσης
πόσα frontier cells αποκλείστηκαν ως μη συνδεδεμένα/μη προσβάσιμα. Έλεγξε στο
RViz τυχόν άγνωστες περιοχές. Ο εξερευνητής παραμένει ενεργός και συνεχίζει αν
εμφανιστούν νέοι στόχοι. Το `UNRESOLVED` παραμένει διακριτό και δεν μετατρέπεται
σε επιτυχία επειδή απέτυχαν όλοι οι στόχοι.

Η ολοκλήρωση **δεν αποθηκεύει αυτόματα** τον χάρτη. Σε νέο τερματικό, αφού κάνεις
τα `source`, αποθήκευσέ τον με μοναδικό όνομα:

```bash
ros2 run nav2_map_server map_saver_cli -f ~/room_map --ros-args -p use_sim_time:=true
```

Αυτό αποθηκεύει τον occupancy map για μελλοντική πλοήγηση, όχι την πλήρη εσωτερική
κατάσταση του Cartographer για συνέχιση του ίδιου SLAM session.

## Παύση, συνέχιση και τερματισμός

Παύση με αίτημα ακύρωσης του ενεργού στόχου:

```bash
ros2 service call /frontier_explorer/pause std_srvs/srv/SetBool '{data: true}'
```

Συνέχιση:

```bash
ros2 service call /frontier_explorer/pause std_srvs/srv/SetBool '{data: false}'
```

Το `Ctrl+C` στον εξερευνητή ζητά ακύρωση και περιμένει έως 5 s. Αν εμφανιστεί
`Cancellation not confirmed`, σταμάτησε και το Nav2 ή πάγωσε το Gazebo. Η απώλεια
του εξερευνητή δεν ακυρώνει αυτόματα έναν στόχο που έχει ήδη δεχτεί το Nav2.

Παρακολούθηση κατάστασης:

```bash
ros2 topic echo /frontier_explorer/status
```

Στο RViz μπορείς να προσθέσεις display τύπου Pose για `/frontier_explorer/goal`.

## Έλεγχος και όρια δοκιμών

```bash
cd ~/Projects/burger/turtlebot3_frontier
python3 -m unittest -v test_frontier
```

Πέρασαν 14 έλεγχοι συνθετικών χαρτών και ρυθμίσεων: ελεύθερες θέσεις στόχων,
κλειστό δωμάτιο, αποσυνδεδεμένο δωμάτιο, φαρδιά/στενή πόρτα, άκυρη θέση ρομπότ,
περιστραμμένη αρχή χάρτη, αποφυγή διαγώνιων συντομεύσεων, cooldown, πεπερασμένες
επισκέψεις και τύποι εντολών Nav2. Ελέγχθηκε επίσης η σύνταξη Python.

**Δεν έγινε πραγματική δοκιμή ROS/Gazebo εδώ.** Οι έλεγχοι δεν πιστοποιούν τη
συμβατότητα των εγκατεστημένων plugins, την αλυσίδα TF, τις ανακτήσεις Nav2 ή την
ποιότητα SLAM στον δικό σου υπολογιστή. Η πρώτη δοκιμή πρέπει να γίνει στο Gazebo.

## Επίσημες αναφορές

- [Nav2: navigation while mapping](https://docs.nav2.org/lyrical/tutorials/general_tutorials/navigation2_with_slam/navigation2_with_slam/)
- [Jazzy navigation-only launch](https://github.com/ros-navigation/navigation2/blob/jazzy/nav2_bringup/launch/navigation_launch.py)
- [NavigateToPose — Jazzy](https://api.nav2.org/actions/jazzy/navigatetopose.html)
- [TurtleBot3 Cartographer simulation](https://emanual.robotis.com/docs/en/platform/turtlebot3/slam_simulation/)
- [Nav2 Jazzy controller server](https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/core_servers/controller_server/)
