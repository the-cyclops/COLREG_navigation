using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine.Splines;


public class BoatAgent : Agent
{
    public HDRPBoatPhysics boatPhysics;
    private Rigidbody rb;
    public GameObject target;

    private float arenaRadius = 15f;

    private float spawnObstacleRadius = 1f;
    private float currentReductionRadius = 10f;

    private int current_step = 0;
    // TO MODIFY WHEN TRAIN STARTS
    private int startSafetyStep = 768_000 * 5; //1 getaction in python corresponds to 5 steps in unity for decisionperiod = 5 

    [SerializeField] GameObject obstacles;
    [SerializeField] GameObject intruderVessel1;
    [SerializeField] GameObject intruderVessel2;

    SplineAnimate splineAnimator1;
    SplineAnimate splineAnimator2;

    [SerializeField] GameObject Path1;
    [SerializeField] GameObject Path2;
    
    private Vector3 initialPosition;
    private Quaternion initialRotation;
    private float previousDistanceToTarget;

    private float intruder1Speed;
    private float intruder2Speed;
    private Vector3 intruder1Velocity;
    private Vector3 intruder2Velocity;

    [SerializeField] private bool debugMode = false;

    public override void Initialize()
    {
        this.MaxStep = 5000; // timer limit for episode, max 5000/100 = 50 s

        boatPhysics = GetComponent<HDRPBoatPhysics>();
        rb = GetComponent<Rigidbody>();

        initialPosition = Vector3.zero;
        initialRotation = Quaternion.identity;

        splineAnimator1 = intruderVessel1.GetComponent<SplineAnimate>();
        splineAnimator2 = intruderVessel2.GetComponent<SplineAnimate>();

        if (debugMode)
        {
            var bp = GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            Debug.Log($"Behavior Name: {bp.BehaviorName}");
            Debug.Log($"Space Size da Inspector: {bp.BrainParameters.VectorObservationSize}");
        }
    }

    void FixedUpdate()
    {
        if (intruderVessel1 != null && intruderVessel1.activeInHierarchy)
        {
            splineAnimator1.ElapsedTime += splineAnimator1.MaxSpeed * Time.fixedDeltaTime;
            intruder1Velocity = intruderVessel1.transform.forward * intruder1Speed;
        }

        if (intruderVessel2 != null && intruderVessel2.activeInHierarchy)
        {
            splineAnimator2.ElapsedTime += splineAnimator2.MaxSpeed * Time.fixedDeltaTime;
            intruder2Velocity = intruderVessel2.transform.forward * intruder2Speed;
        }
    }

    private bool CheckTargetPosition(Vector2 pos)
    {
        float minDistance = 1.5f + spawnObstacleRadius; // Minimum distance from obstacles

        foreach (Transform obstacle in obstacles.transform)
        {
            if (debugMode)
            {
                Debug.Log($"Checking obstacle {obstacle.name}");
            }
            Vector3 obstaclePos = obstacle.transform.localPosition;
            float distance = Vector2.Distance(new Vector2(obstaclePos.x, obstaclePos.z), pos);
            if (distance < minDistance)
            {
                return false; // Too close to an obstacle
            }
        }
        return true; // Valid position
    }   


    private void MoveTarget()
    {
        // Set the curriculum radius based on the training step
        // initially currentReductionRadius is 13
        if (current_step >= startSafetyStep / 2)
        {
            if (current_step >= startSafetyStep)
            {
                currentReductionRadius = 1f; 
                
            }
            else
            {
                currentReductionRadius = 5f;
            }
        }
        else
        {
            currentReductionRadius = 10f;
        }
        
        float maxSpawnDist = arenaRadius - currentReductionRadius;
        float minSpawnDist = 2f + spawnObstacleRadius; // Minimum distance from target
        // Randomly sample a point within a donut-shaped area 
        // bounded by minSpawnDist and maxSpawnDist to prevent overlapping with target at spawn
        Vector2 randomPoint;
        do
        {
            //randomCircle = UnityEngine.Random.insideUnitCircle * (arenaRadius-currentReductionRadius);
            Vector2 randomDir = UnityEngine.Random.insideUnitCircle.normalized;
            float randomDist = UnityEngine.Random.Range(minSpawnDist, maxSpawnDist);
            randomPoint = randomDir * randomDist;
        }
        while (!CheckTargetPosition(randomPoint));

        Vector3 targetPosition = new Vector3(randomPoint.x, 0.0f, randomPoint.y);
        target.transform.localPosition = targetPosition;
    }

    private void MoveObstacles()
    {
        foreach (Transform obstacle in obstacles.transform)
        {
            Vector2 randomCircle = UnityEngine.Random.insideUnitCircle * spawnObstacleRadius;
            Vector3 newPos = new Vector3(randomCircle.x, 0.0f, randomCircle.y);
            Transform sphereTransform = obstacle.Find("Sphere");
            if (sphereTransform != null)
            {
                GameObject obstacleObj = sphereTransform.gameObject;
                obstacleObj.transform.localPosition = newPos;
            }
        }
    }

    private void MoveIntruders()
    {
        // ---Path 1---
        float scaleX1 = UnityEngine.Random.Range(0.9f, 1.1f);
        float scaleZ1 = UnityEngine.Random.Range(0.9f, 1.1f);
        Path1.transform.localScale = new Vector3(scaleX1, 1f, scaleZ1);

        // Definiamo la velocità nel mondo (es. 2.3 m/s)
        intruder1Speed = UnityEngine.Random.Range(2.1f, 2.5f);
        float maxScale1 = Mathf.Max(scaleX1, scaleZ1);

        // Impostiamo ElapsedTime a un punto casuale senza bias
        splineAnimator1.ElapsedTime = UnityEngine.Random.Range(0f, splineAnimator1.Duration);

        // Calcoliamo la velocità locale per far sì che la velocità mondo sia corretta
        // Usiamo MaxSpeed come contenitore per il calcolo nel FixedUpdate
        splineAnimator1.MaxSpeed = intruder1Speed / maxScale1;

        // --- Path 2---
        float scaleX2 = UnityEngine.Random.Range(0.9f, 1.1f);
        float scaleZ2 = UnityEngine.Random.Range(0.9f, 1.1f);
        Path2.transform.localScale = new Vector3(scaleX2, 1f, scaleZ2);

        intruder2Speed = UnityEngine.Random.Range(2.1f, 2.5f);
        float maxScale2 = Mathf.Max(scaleX2, scaleZ2);

        splineAnimator2.ElapsedTime = UnityEngine.Random.Range(0f, splineAnimator2.Duration);
        splineAnimator2.MaxSpeed = intruder2Speed / maxScale2;

        if (debugMode)
        {
            Debug.Log($"Intruder 1 World Speed: {intruder1Speed} | Local MaxSpeed: {splineAnimator1.MaxSpeed}");
            Debug.Log($"Intruder 2 World Speed: {intruder2Speed} | Local MaxSpeed: {splineAnimator2.MaxSpeed}");
        }
    }

    public override void OnEpisodeBegin()
    {
        
        // Move Obstacles
        MoveObstacles();

        // Move Intruders
        MoveIntruders();

        // Move Target 
        MoveTarget();

        //Reset Boat Velocities
        boatPhysics.ResetVelocities();

        // Reset Boat Position and Rotation
        transform.localPosition = initialPosition;
        transform.localRotation = initialRotation;

        Physics.SyncTransforms();

        intruder1Velocity = Vector3.zero;
        intruder2Velocity = Vector3.zero;

        Vector2 boatPos2D = new Vector2(transform.localPosition.x, transform.localPosition.z);
        Vector2 targetPos2D = new Vector2(target.transform.localPosition.x, target.transform.localPosition.z);
        previousDistanceToTarget = Vector2.Distance(boatPos2D, targetPos2D);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        //TODO CHECK THE ORDER OF OBSERVATIONS - IT MATTERS?

        // Observation Structure:
        // 0-2: Target Relative Position (3)
        // 3:   Target Distance (1)
        // 4-6: Linear Velocity (3)
        // 7-9: Angular Velocity (3)
        // 10-16: Intruder Vessel 1 - Position, Distance, Relative Velocity (7)
        // 17-23: Intruder Vessel 2 - Position, Distance, Relative Velocity (7)

        // --- SELF & TARGET OBSERVATIONS ---

        // Fetch Target Position Relative to Boat
        Vector3 targetRelativePos = transform.InverseTransformPoint(target.transform.position);
        if (debugMode) Debug.Log("Target Relative Position: " + targetRelativePos.ToString("F2"));
        // Obs Index [0,1,2]: Target Relative Position (Local space) - Direction to the target
        sensor.AddObservation(targetRelativePos.normalized);

        // Fetch Target Distance
        float targetDistance = targetRelativePos.magnitude;
        // Normalized Distance (Assuming max distance of 20 units) 
        // Rational normalization d/(d+k) with k=15. Range: [0, 1]
        // Obs Index [3]: Target Distance - How far is the target
        if (debugMode) Debug.Log("Target Distance: " + targetDistance.ToString("F2"));
        sensor.AddObservation(targetDistance / (arenaRadius + targetDistance));

        // Fetch Boat Velocity
        // InverseTransformDirection converts world space velocity to local space
        Vector3 localLinearVelocity = transform.InverseTransformDirection(rb.linearVelocity);
        if (debugMode) Debug.Log("Local Linear Velocity: " + localLinearVelocity.ToString("F2"));
        Vector3 normalizedLocalLinearVelocity = localLinearVelocity / boatPhysics.nominalMaxLinearSpeed;
        // Clamp magnitude to in -1 to 1 range
        // Obs Index [4,5,6]: Linear Velocity (Local space) - LV of the boat
        sensor.AddObservation(Vector3.ClampMagnitude(normalizedLocalLinearVelocity, 1.0f));

        // Fetch Boat Angular Velocity
        Vector3 localAngularVelocity = transform.InverseTransformDirection(rb.angularVelocity);
        Vector3 normalizedLocalAngularVelocity = localAngularVelocity / boatPhysics.nominalMaxAngularSpeed;
        if (debugMode) Debug.Log("Local Angular Velocity: " + localAngularVelocity.ToString("F2"));
        // Clamp magnitude to in -1 to 1 range
        // Obs Index [7,8,9]: Angular Velocity (Local space) - AV of the boat
        sensor.AddObservation(Vector3.ClampMagnitude(normalizedLocalAngularVelocity, 1.0f));

        // --- COLREG / INTRUDER VESSEL OBSERVATIONS ---
        // Necessary for calculating CPA and Velocity Obstacles in Python
    
        // INTRUDER VESSEL 1
        // If no vessel is present, you should feed zeros or a specific flag
        if (intruderVessel1 != null && intruderVessel1.activeInHierarchy)
        {
            // 1. Relative Position of the first intruder vessel (Local space)
            Vector3 intruder1RelativePos = transform.InverseTransformPoint(intruderVessel1.transform.position);
            if (debugMode) Debug.Log("Intruder 1 Relative Position: " + intruder1RelativePos.ToString("F2"));
            // Obs Index [10,11,12]: Intruder 1 Relative Position (Local space)
            sensor.AddObservation(intruder1RelativePos.normalized); 
        
            float intruder1Dist = intruder1RelativePos.magnitude;
            // Obs Index [13]: Intruder 1 Distance
            if (debugMode) Debug.Log("Intruder 1 Distance: " + intruder1Dist.ToString("F2"));
            sensor.AddObservation(intruder1Dist / (arenaRadius + intruder1Dist)); // Normalized Distance

            // 2. Relative Velocity (Crucial for CPA/Collision Risk)
            // We calculate the vector difference in world space, then convert to local
            Vector3 relativeVelocityWorld1 = intruder1Velocity - rb.linearVelocity;
            Vector3 localRelativeVelocity1 = transform.InverseTransformDirection(relativeVelocityWorld1);
            // Normalize by 2 * max speed in order to distinguish between one vessell full speed or both at full speed
            // Obs Index [14,15,16]: Intruder 1 Relative Velocity (Local space)
            if (debugMode) Debug.Log("Intruder 1 Relative Velocity: " + localRelativeVelocity1.ToString("F2"));
            sensor.AddObservation(Vector3.ClampMagnitude(localRelativeVelocity1 / (boatPhysics.nominalMaxLinearSpeed * 2f), 1.0f));   
        }
        else
        {
            // Padding if no intruder is active to keep observation size constant
            // Obs Index [10-16]: Zeros for Intruder 1
            sensor.AddObservation(Vector3.zero); // Rel Pos
            sensor.AddObservation(1.0f);           // Dist
            sensor.AddObservation(Vector3.zero); // Rel Vel
        }

        // INTRUDER VESSEL 2
        // If no vessel is present, you should feed zeros or a specific flag
        if (intruderVessel2 != null && intruderVessel2.activeInHierarchy)
        {
            // 1. Relative Position of the second intruder vessel (Local space)
            Vector3 intruder2RelativePos = transform.InverseTransformPoint(intruderVessel2.transform.position);
            if (debugMode) Debug.Log("Intruder 2 Relative Position: " + intruder2RelativePos.ToString("F2"));
            // Obs Index [17,18,19]: Intruder 2 Relative Position (Local space)
            sensor.AddObservation(intruder2RelativePos.normalized); 
        
            float intruder2Dist = intruder2RelativePos.magnitude;
            if (debugMode) Debug.Log("Intruder 2 Distance: " + intruder2Dist.ToString("F2"));
            // Obs Index [20]: Intruder 2 Distance
            sensor.AddObservation(intruder2Dist / (arenaRadius + intruder2Dist)); // Normalized Distance

            // 2. Relative Velocity (Crucial for CPA/Collision Risk)
            // We calculate the vector difference in world space, then convert to local
            Vector3 relativeVelocityWorld2 = intruder2Velocity - rb.linearVelocity;
            Vector3 localRelativeVelocity2 = transform.InverseTransformDirection(relativeVelocityWorld2);
            // Normalize by 2 * max speed in order to distinguish between one vessell full speed or both at full speed
            // Obs Index [21,22,23]: Intruder 2 Relative Velocity (Local space)
            if (debugMode) Debug.Log("Intruder 2 Relative Velocity: " + localRelativeVelocity2.ToString("F2"));
            sensor.AddObservation(Vector3.ClampMagnitude(localRelativeVelocity2 / (boatPhysics.nominalMaxLinearSpeed * 2f), 1.0f));
        }
        else
        {
            // Padding if no intruder is active to keep observation size constant
            // Obs Index [17-23]: Zeros for Intruder 2
            sensor.AddObservation(Vector3.zero); // Rel Pos
            sensor.AddObservation(1.0f);           // Dist
            sensor.AddObservation(Vector3.zero); // Rel Vel
        }

    // Total Observations: 3 (targetRelativePos) + 1 (targetDistance) + 3 (linearVelocity) + 3 (angularVelocity) + 3 (intruder1RelativePos) + 1 (intruder1Dist) + 3 (intruder1RelVel) + 3 (intruder2RelativePos) + 1 (intruder2Dist) + 3 (intruder2RelVel) = 24
    // // Adjust the Space Size in Vector Sensor in Behavior Parameters accordingly 
    }

    // This method is called every step during training OR by your keyboard in Heuristic mode
    // the agent will have a maximum number of 9k step per episode
    public override void OnActionReceived(ActionBuffers actions)
    {
        var continuousActions = actions.ContinuousActions;

        // L2 Energy Penalty
        AddReward(-0.001f * ((continuousActions[0] * continuousActions[0]) + (continuousActions[1] * continuousActions[1])));
        // L1 Energy Penalty
        //AddReward(-0.001f * (Mathf.Abs(continuousActions[0]) + Mathf.Abs(continuousActions[1])));
        
        // Differential Thrust Mapping
        //float leftInput = Mathf.Clamp(continuousActions[0], -1f, 1f);
        //float rightInput = Mathf.Clamp(continuousActions[1], -1f, 1f);

        // Differential Drive Mixer
        float throttle = Mathf.Clamp(continuousActions[0], -1f, 1f);
        float steering = Mathf.Clamp(continuousActions[1], -1f, 1f);
        float leftInput = throttle + steering;
        float rightInput = throttle - steering;

        float maxInput = Mathf.Max(Mathf.Abs(leftInput), Mathf.Abs(rightInput));
        // Normalize inputs if any exceeds the range [-1, 1] to maintain the intended throttle/steering ratio
        if (maxInput > 1f)
        {
            leftInput /= maxInput;
            rightInput /= maxInput;
        }
        // Apply forces based on jet inputs
        boatPhysics.SetJetInputs(leftInput, rightInput);

        // Reward to incentivize getting closer to the target
        Vector2 boatPos2D = new Vector2(transform.localPosition.x, transform.localPosition.z);
        Vector2 targetPos2D = new Vector2(target.transform.localPosition.x, target.transform.localPosition.z);
        float currentDistanceToTarget = Vector2.Distance(boatPos2D, targetPos2D);
        float distanceReward = previousDistanceToTarget - currentDistanceToTarget; 
        AddReward(distanceReward * 1f); // Scale the reward for distance improvement
        previousDistanceToTarget = currentDistanceToTarget;

        // Reward to incetivize mantainig direction and speed towards the target
        Vector3 dirToTarget = (target.transform.position - transform.position).normalized;
        float facingTarget = Vector3.Dot(transform.forward, dirToTarget);
        AddReward(facingTarget * 0.0005f);

        // Small penalty to reduce rotation on the spot
        AddReward(-0.005f * Mathf.Abs(rb.angularVelocity.y));

        // Time penalty
        AddReward(-5.0f / MaxStep);  
        current_step++;
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (!collision.gameObject.CompareTag("Target")) {
            // same penalty for all collisons as professor suggested
            // max penalty of being alive for maxsteps 
            if (collision.gameObject.CompareTag("Obstacle") || collision.gameObject.CompareTag("Wall") || collision.gameObject.CompareTag("Boat")) {
                AddReward(-6.0f);
            }
            if (debugMode) Debug.Log(GetCumulativeReward());
            EndEpisode();
        }

    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Target"))
        {
            AddReward(15.0f);
            // 2. Coloriamo il pavimento di verde per feedback visivo (Opzionale ma bello)
            // StartCoroutine(SwapGroundMaterial(successMaterial, 0.5f));

            if (debugMode) Debug.Log(GetCumulativeReward());
            EndEpisode(); 
        }
    }

    // This maps your keyboard to the Action Buffers
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Get reference to the continuous actions
        var continuousActions = actionsOut.ContinuousActions;
        
        // Vertical axis (W/S or Up/Down) -> Throttle
        float forwardInput = Input.GetAxis("Vertical"); 
        
        // Horizontal axis (A/D or Left/Right) -> Steering
        float turnInput = Input.GetAxis("Horizontal");


        //Mixing Commands for Jet Engines
        // By pressing W both jets push forward (+,+)
        // By pressing S both jets push backward (-,-)
        // By pressing A left jet backward, right jet forward (-,+)
        // By pressing D left jet forward, right jet backward (+,-)
        float leftJet = forwardInput + turnInput;
        float rightJet = forwardInput - turnInput;


        // Clamp (clip) the values between -1 and 1
        continuousActions[0] = Mathf.Clamp(leftJet, -1f, 1f);
        continuousActions[1] = Mathf.Clamp(rightJet, -1f, 1f);
    }

}