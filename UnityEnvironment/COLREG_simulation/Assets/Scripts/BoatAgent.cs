using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine.Splines;
using Unity.VisualScripting;


public class BoatAgent : Agent
{
    public HDRPBoatPhysics boatPhysics;
    private Rigidbody rb;
    public GameObject target;

    private float arenaRadius = 15f;
    private float maxDistance = 43f;
    private float spawnObstacleRadius = 1f;
    private float currentReductionRadius = 10f;

    private float spawnDistance;
    private float invSpawnDistance;

    // TEST COUNTER
    private int currentEpisodeStep = 0;


    private float minSpawnDist = 3f; // Minimum distance from target

    private int current_step = 0;
    //private int startSafetyStep = 1_024_000 * 5; //1 getaction in python corresponds to 5 steps in unity for decisionperiod = 5 

    private int curriculumStage = 0; // 0: Empty Arena, 1: Fixed Obstacles, 2: Moving Obstacles

    private int stage1Threshold = 251_904 * 5; // Update 123
    private int stage2Threshold = 501_760 * 5; // Update 245

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

    // --- VARIABILI AGGIUNTE PER SANITY CHECK (XZ) ---
    private Vector2 lastPosIntruder1_2D;
    private Vector2 lastPosIntruder2_2D;
    private float realSpeedIntruder1;
    private float realSpeedIntruder2;

    [SerializeField] private bool debugMode = false;

    private System.Random random;
    private System.Random trainRandom;
    private System.Random evalRandom;

    private float getRandomFloat(float min, float max)
    {
        return (float)(random.NextDouble() * (max - min) + min);
    }

    private Vector2 getRandomUniformInCircle(float radius)
    {
        double angle = random.NextDouble() * 2 * Mathf.PI;
        double r = radius * Mathf.Sqrt((float)random.NextDouble());
        return new Vector2((float)(r * Mathf.Cos((float)angle)), (float)(r * Mathf.Sin((float)angle)));
    }

    public override void Initialize()
    {
        this.MaxStep = 5000; // timer limit for episode, max 5000/100 = 50 s

        //int trainSeed = (int)Academy.Instance.EnvironmentParameters.GetWithDefault("train_seed", 0);
        //int evalSeed = (int)Academy.Instance.EnvironmentParameters.GetWithDefault("eval_seed", 0);
        //trainRandom = new System.Random(trainSeed);
        //evalRandom = new System.Random(evalSeed);
        int seed = (int)Academy.Instance.EnvironmentParameters.GetWithDefault("seed", 0);
        random = new System.Random(seed);

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
        if (intruderVessel1 != null && intruderVessel1.activeInHierarchy && curriculumStage == 2)
    {
        // Usiamo il valore teorico intruder1Speed per l'IA
        intruder1Velocity = intruderVessel1.transform.forward * intruder1Speed;
    }

    if (intruderVessel2 != null && intruderVessel2.activeInHierarchy && curriculumStage == 2)
    {
        intruder2Velocity = intruderVessel2.transform.forward * intruder2Speed;
    }

        // Print di controllo (Sanity Check)
        if (debugMode && Time.frameCount % 100 == 0)
        {
            float agentSpeedXZ = new Vector2(rb.linearVelocity.x, rb.linearVelocity.z).magnitude;
            Debug.Log($"[CHECK XZ] Agente: {agentSpeedXZ:F2} | I1: {realSpeedIntruder1:F2} | I2: {realSpeedIntruder2:F2}");
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
        if (curriculumStage == 2) // 10 to 14
        {
            currentReductionRadius = 1f;
            minSpawnDist = 10f;
        }
        else if (curriculumStage == 1) // 8 to 12
        {
            currentReductionRadius = 3f;
            minSpawnDist = 8f;
        }
        else // 6 to 10
        {
            currentReductionRadius = 5f;
            minSpawnDist = 6f;
        }
        
        float maxSpawnDist = arenaRadius - currentReductionRadius;
        // Randomly sample a point within a donut-shaped area 
        // bounded by minSpawnDist and maxSpawnDist to prevent overlapping with target at spawn
        Vector2 randomPoint;
        do
        {
            //randomCircle = UnityEngine.Random.insideUnitCircle * (arenaRadius-currentReductionRadius);
            Vector2 randomDir = getRandomUniformInCircle(1f).normalized; // Random direction
            float randomDist = getRandomFloat(minSpawnDist, maxSpawnDist);
            randomPoint = randomDir * randomDist;
        }
        while (!CheckTargetPosition(randomPoint));

        Vector3 targetPosition = new Vector3(randomPoint.x, 0.0f, randomPoint.y);
        target.transform.localPosition = targetPosition;
        spawnDistance = targetPosition.magnitude;
         
    }

    private void MoveObstacles()
    {

        float radius = 0f;

        if (curriculumStage == 0)
        {
            obstacles.SetActive(false);
            return;
        }
        else
        {
            obstacles.SetActive(true);
            if (curriculumStage == 1)
            {
                radius = 0f; // Fixed Obstacles, no randomization
            } else
            {
                radius = spawnObstacleRadius;    
            }
            
        }

        foreach (Transform obstacle in obstacles.transform)
        {
            Vector2 randomCircle = getRandomUniformInCircle(radius);
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
    if (curriculumStage < 2)
    {
        intruderVessel1.SetActive(false);
        intruderVessel2.SetActive(false);
        return;
    }

    // Riattiviamo entrambi
    intruderVessel1.SetActive(true);
    intruderVessel2.SetActive(true);

    // Range di velocità richiesto
    float minS = 1.9f;
    float maxS = 2.2f;

    // --- Path 1 Setup ---
    float scaleX1 = getRandomFloat(0.9f, 1.1f);
    float scaleZ1 = getRandomFloat(0.9f, 1.1f);
    Path1.transform.localScale = new Vector3(scaleX1, 1f, scaleZ1);

    intruder1Speed = getRandomFloat(minS, maxS);
    // Compensazione: velocità locale = velocità desiderata / scala massima del percorso
    splineAnimator1.MaxSpeed = intruder1Speed / Mathf.Max(scaleX1, scaleZ1);
    
    // Partenza casuale lungo il percorso per non avere bias di posizione
    splineAnimator1.ElapsedTime = getRandomFloat(0f, splineAnimator1.Duration);
    splineAnimator1.Play(); 

    // --- Path 2 Setup ---
    float scaleX2 = getRandomFloat(0.9f, 1.1f);
    float scaleZ2 = getRandomFloat(0.9f, 1.1f);
    Path2.transform.localScale = new Vector3(scaleX2, 1f, scaleZ2);

    intruder2Speed = getRandomFloat(minS, maxS);
    splineAnimator2.MaxSpeed = intruder2Speed / Mathf.Max(scaleX2, scaleZ2);
    
    splineAnimator2.ElapsedTime = getRandomFloat(0f, splineAnimator2.Duration);
    splineAnimator2.Play(); 

    if (debugMode)
    {
        Debug.Log($"[SPAWN] I1 Speed: {intruder1Speed:F2} | I2 Speed: {intruder2Speed:F2}");
    }
}

    public override void OnEpisodeBegin()
    {
        float evalEpisodeSeed = Academy.Instance.EnvironmentParameters.GetWithDefault("eval_episode_seed", -1f);
        if (evalEpisodeSeed != -1f)
        {
            random = new System.Random((int)evalEpisodeSeed);
            curriculumStage = 2; 
        }

        //float evalMode = Academy.Instance.EnvironmentParameters.GetWithDefault("eval_mode", 0);
        //random = evalMode > 0 ? evalRandom : trainRandom;

        // TEST COUNTER
        currentEpisodeStep = 0;
        
        // Move Obstacles
        MoveObstacles();

        // Move Intruders
        MoveIntruders();

        // Move Target 
        MoveTarget();
        invSpawnDistance = 1f / Mathf.Max(spawnDistance, 1e-4f);

        //Reset Boat Velocities
        boatPhysics.ResetVelocities();

        // Reset Boat Position and Rotation
        transform.localPosition = initialPosition;
        transform.localRotation = initialRotation;

        Physics.SyncTransforms();

        // RESET POSIZIONI PER EVITARE SPIKE DI VELOCITÀ AL PRIMO FRAME
        lastPosIntruder1_2D = new Vector2(intruderVessel1.transform.position.x, intruderVessel1.transform.position.z);
        lastPosIntruder2_2D = new Vector2(intruderVessel2.transform.position.x, intruderVessel2.transform.position.z);

        intruder1Velocity = Vector3.zero;
        intruder2Velocity = Vector3.zero;

        Vector2 boatPos2D = new Vector2(transform.localPosition.x, transform.localPosition.z);
        Vector2 targetPos2D = new Vector2(target.transform.localPosition.x, target.transform.localPosition.z);
        previousDistanceToTarget = Vector2.Distance(boatPos2D, targetPos2D);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Observation Structure:
        // 0-1: Target Relative Position (2) - X and Z in local space (direction to the target)
        // 2:   Target Distance (1)
        // 3-4: Linear Velocity (2) - X and Z in local space
        // 5: Angular Velocity (1) - Yaw (Y) in local space
        // 6-10: Intruder Vessel 1 - Position, Distance, Relative Velocity (5) = (2 + 1 + 2)
        // 11-12: Intruder Vessel 1 - Heading (2) - Direction in local space
        // 13-17: Intruder Vessel 2 - Position, Distance, Relative Velocity (5) = (2 + 1 + 2)
        // 18-19: Intruder Vessel 2 - Heading (2) - Direction in local space

        // --- SELF & TARGET OBSERVATIONS ---

        // Fetch Target Position Relative to Boat
        Vector3 targetRelativePos = transform.InverseTransformPoint(target.transform.position);
        Vector2 targetRelativePos2D = new Vector2(targetRelativePos.x, targetRelativePos.z);
        if (debugMode) Debug.Log("Target Relative Position: " + targetRelativePos2D.ToString("F2"));
        // Obs Index [0,1]: Target Relative Position (Local space) - Direction to the target
        sensor.AddObservation(targetRelativePos2D.normalized);

        // Fetch Target Distance
        float targetDistance = new Vector2(targetRelativePos.x, targetRelativePos.z).magnitude; // Calcolo XZ
        // Normalized Distance using maxDistance. Range: [0, 1]
        // Obs Index [2]: Target Distance - How far is the target
        if (debugMode) Debug.Log("Target Distance: " + targetDistance.ToString("F2"));
        sensor.AddObservation(Mathf.Clamp01(targetDistance / maxDistance));

        // Fetch Boat Velocity
        // InverseTransformDirection converts world space velocity to local space
        Vector3 localLinearVelocity = transform.InverseTransformDirection(rb.linearVelocity);
        Vector2 localLinearVelocity2D = new Vector2(localLinearVelocity.x, localLinearVelocity.z);
        if (debugMode) Debug.Log("Local Linear Velocity: " + localLinearVelocity.ToString("F2"));
        Vector2 normalizedLocalLinearVelocity2D = localLinearVelocity2D / boatPhysics.nominalMaxLinearSpeed;
        // Clamp magnitude to in -1 to 1 range
        // Obs Index [3,4]: Linear Velocity (Local space) - LV of the boat
        sensor.AddObservation(Vector2.ClampMagnitude(normalizedLocalLinearVelocity2D, 1.0f));

        // Fetch Boat Angular Velocity
        Vector3 localAngularVelocity = transform.InverseTransformDirection(rb.angularVelocity);
        float localAngularVelocityY = localAngularVelocity.y; 
        float normalizedLocalAngularVelocity = localAngularVelocityY / boatPhysics.nominalMaxAngularSpeed;
        if (debugMode) Debug.Log("Local Angular Velocity: " + localAngularVelocityY.ToString("F2"));
        // Clamp magnitude to in -1 to 1 range
        // Obs Index [5]: Angular Velocity (Local space) - AV of the boat
        sensor.AddObservation(normalizedLocalAngularVelocity);

        // --- COLREG / INTRUDER VESSEL OBSERVATIONS ---
        // Necessary for calculating CPA and Velocity Obstacles in Python
    
        // INTRUDER VESSEL 1
        // If no vessel is present, you should feed zeros or a specific flag
        if (intruderVessel1 != null && intruderVessel1.activeInHierarchy)
        {
            // 1. Relative Position of the first intruder vessel (Local space)
            Vector3 intruder1RelativePos = transform.InverseTransformPoint(intruderVessel1.transform.position);
            Vector2 intruder1RelativePos2D = new Vector2(intruder1RelativePos.x, intruder1RelativePos.z);
            if (debugMode) Debug.Log("Intruder 1 Relative Position: " + intruder1RelativePos2D.ToString("F2"));
            // Obs Index [6,7]: Intruder 1 Relative Position (Local space)
            sensor.AddObservation(intruder1RelativePos2D.normalized); 
        
            float intruder1Dist = intruder1RelativePos2D.magnitude; // XZ
            // Obs Index [8]: Intruder 1 Distance
            if (debugMode) Debug.Log("Intruder 1 Distance: " + intruder1Dist.ToString("F2"));
            sensor.AddObservation(Mathf.Clamp01(intruder1Dist / maxDistance)); // Normalized Distance

            // 2. Relative Velocity (Crucial for CPA/Collision Risk)
            // Usiamo la velocità XZ calcolata nel FixedUpdate
            Vector3 relativeVelocityWorld1 = intruder1Velocity - new Vector3(rb.linearVelocity.x, 0, rb.linearVelocity.z);
            Vector3 localRelativeVelocity1 = transform.InverseTransformDirection(relativeVelocityWorld1);
            Vector2 localRelativeVelocity1_2D = new Vector2(localRelativeVelocity1.x, localRelativeVelocity1.z);
            // Normalize by 2 * max speed in order to distinguish between one vessell full speed or both at full speed
            // Obs Index [9,10]: Intruder 1 Relative Velocity (Local space)
            if (debugMode) Debug.Log("Intruder 1 Relative Velocity: " + localRelativeVelocity1_2D.ToString("F2"));
            if (debugMode) Debug.Log("Observation intruder 1 Rel Vel (Normalized): " + Vector2.ClampMagnitude(localRelativeVelocity1_2D / (boatPhysics.nominalMaxLinearSpeed * 2f), 1.0f).ToString("F2"));
            sensor.AddObservation(Vector2.ClampMagnitude(localRelativeVelocity1_2D / (boatPhysics.nominalMaxLinearSpeed * 2f), 1.0f));
            
            // 3. Heading of the intruder vessel (Direction in local space)
            Vector3 intruder1HeadingLocal = transform.InverseTransformDirection(intruderVessel1.transform.forward);
            Vector2 intruder1Heading2D = new Vector2(intruder1HeadingLocal.x, intruder1HeadingLocal.z).normalized;
            // Obs Index [11,12]: Intruder 1 Heading (Local space)
            if (debugMode) Debug.Log("Intruder 1 Heading: " + intruder1Heading2D.ToString("F2"));
            sensor.AddObservation(intruder1Heading2D);
        }
        else
        {
            // Padding if no intruder is active to keep observation size constant
            // Obs Index [6-12]: Zeros for Intruder 1
            sensor.AddObservation(Vector2.zero); // Rel Pos
            sensor.AddObservation(1.0f);           // Dist
            sensor.AddObservation(Vector2.zero); // Rel Vel
            sensor.AddObservation(Vector2.zero); // Heading
        }

        // INTRUDER VESSEL 2
        // If no vessel is present, you should feed zeros or a specific flag
        if (intruderVessel2 != null && intruderVessel2.activeInHierarchy)
        {
            // 1. Relative Position of the second intruder vessel (Local space)
            Vector3 intruder2RelativePos = transform.InverseTransformPoint(intruderVessel2.transform.position);
            Vector2 intruder2RelativePos2D = new Vector2(intruder2RelativePos.x, intruder2RelativePos.z);
            if (debugMode) Debug.Log("Intruder 2 Relative Position: " + intruder2RelativePos2D.ToString("F2"));
            // Obs Index [13,14]: Intruder 2 Relative Position (Local space)
            sensor.AddObservation(intruder2RelativePos2D.normalized); 
        
            float intruder2Dist = intruder2RelativePos2D.magnitude; // XZ
            if (debugMode) Debug.Log("Intruder 2 Distance: " + intruder2Dist.ToString("F2"));
            // Obs Index [15]: Intruder 2 Distance
            sensor.AddObservation(Mathf.Clamp01(intruder2Dist / maxDistance)); // Normalized Distance

            // 2. Relative Velocity (Crucial for CPA/Collision Risk)
            Vector3 relativeVelocityWorld2 = intruder2Velocity - new Vector3(rb.linearVelocity.x, 0, rb.linearVelocity.z);
            Vector3 localRelativeVelocity2 = transform.InverseTransformDirection(relativeVelocityWorld2);
            Vector2 localRelativeVelocity2_2D = new Vector2(localRelativeVelocity2.x, localRelativeVelocity2.z);
            // Normalize by 2 * max speed in order to distinguish between one vessell full speed or both at full speed
            // Obs Index [16,17]: Intruder 2 Relative Velocity (Local space)
            if (debugMode) Debug.Log("Intruder 2 Relative Velocity: " + localRelativeVelocity2_2D.ToString("F2"));
            if (debugMode) Debug.Log("Observation intruder 2 Rel Vel (Normalized): " + Vector2.ClampMagnitude(localRelativeVelocity2_2D / (boatPhysics.nominalMaxLinearSpeed * 2f), 1.0f).ToString("F2"));
            sensor.AddObservation(Vector2.ClampMagnitude(localRelativeVelocity2_2D / (boatPhysics.nominalMaxLinearSpeed * 2f), 1.0f));
            
            // 3. Heading of the intruder vessel (Direction in local space)
            Vector3 intruder2HeadingLocal = transform.InverseTransformDirection(intruderVessel2.transform.forward);
            Vector2 intruder2Heading2D = new Vector2(intruder2HeadingLocal.x, intruder2HeadingLocal.z).normalized;
            // Obs Index [18,19]: Intruder 2 Heading (Local space)
            if (debugMode) Debug.Log("Intruder 2 Heading: " + intruder2Heading2D.ToString("F2"));
            sensor.AddObservation(intruder2Heading2D);
        }
        else
        {
            // Padding if no intruder is active to keep observation size constant
            // Obs Index [13-19]: Zeros for Intruder 2
            sensor.AddObservation(Vector2.zero); // Rel Pos
            sensor.AddObservation(1.0f);           // Dist
            sensor.AddObservation(Vector2.zero); // Rel Vel
            sensor.AddObservation(Vector2.zero); // Heading
        }
    }

    // This method is called every step during training OR by your keyboard in Heuristic mode
    public override void OnActionReceived(ActionBuffers actions)
    {
        var continuousActions = actions.ContinuousActions;
        float stepReward = 0f;

        // Differential Drive Mixer
        float throttle = Mathf.Clamp(continuousActions[0], -1f, 1f);
        float steering = Mathf.Clamp(continuousActions[1], -1f, 1f);

        float leftInput = throttle + steering;
        float rightInput = throttle - steering;

        float maxInput = Mathf.Max(Mathf.Abs(leftInput), Mathf.Abs(rightInput));
        // Normalize inputs if any exceeds the range [-1, 1]
        if (maxInput > 1f)
        {
            leftInput /= maxInput;
            rightInput /= maxInput;
        }
        // Apply forces based on jet inputs
        boatPhysics.SetJetInputs(leftInput, rightInput);

        Vector2 boatPos2D = new Vector2(transform.localPosition.x, transform.localPosition.z);
        Vector2 targetPos2D = new Vector2(target.transform.localPosition.x, target.transform.localPosition.z);
        float currentDistanceToTarget = Vector2.Distance(boatPos2D, targetPos2D);
        // Reward to incentivize getting closer to the target
        float distanceReward = previousDistanceToTarget - currentDistanceToTarget; //15 -12 = +3 (good) | 15 -18 = -3 (bad)
        distanceReward *= invSpawnDistance; // Normalizziamo per la distanza di spawn per mantenere coerenza del reward indipendentemente da dove appare il target

        // TEST COUNTER - DA COMMENTARE PER TEST
        //distanceReward = distanceReward * (1f - (currentEpisodeStep / MaxStep)); // Decay del reward di distanza

        currentEpisodeStep++;


        previousDistanceToTarget = currentDistanceToTarget;

        Vector3 dirToTarget = target.transform.position - transform.position;
        dirToTarget.y = 0; // XZ Only
        float facingTarget = Vector3.Dot(transform.forward, dirToTarget.normalized);

        // TEST NEGATIVE REWARD FOR FACING AWAY FROM TARGET
        if (facingTarget < 0 && distanceReward > 0)
        {
            distanceReward *= 0.5f;
        }
        stepReward += distanceReward * 1f; // Scale the reward for distance improvement
        // small encoragment to face correcyly
        //if (facingTarget > 0)
        //{
        //    AddReward(facingTarget * 0.0001f);
        //}
        if (facingTarget > 0.8)
        {
        // Reward to incetivize mantainig direction and speed towards the target
            stepReward += facingTarget * 0.0001f;
        }
        // possible penalty for reverse
        Vector3 flatForward = transform.forward;
        flatForward.y = 0;
        flatForward.Normalize();
        Vector3 flatVelocity = rb.linearVelocity;
        flatVelocity.y = 0;
        float forwardSpeed = Vector3.Dot(flatForward, flatVelocity);
        if (forwardSpeed < -0.1f)
        {
            stepReward += forwardSpeed * 0.0001f;
        }
        // penalty to maintain stability
        stepReward += -0.00005f * Mathf.Abs(rb.angularVelocity.y);
        // Time penalty
        stepReward += -10.0f / MaxStep;

        AddReward(stepReward); 
 
        current_step++;
        if (current_step == stage1Threshold)
        {
            curriculumStage = 1;
            if (debugMode) Debug.Log("Curriculum Stage 1: Fixed Obstacles");
        }
        else if (current_step == stage2Threshold)
        {
            curriculumStage = 2;
            if (debugMode) Debug.Log("Curriculum Stage 2: Moving Obstacles");
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (!collision.gameObject.CompareTag("Target")) {
            if (collision.gameObject.CompareTag("Obstacle") || collision.gameObject.CompareTag("Wall") || collision.gameObject.CompareTag("Boat")) {
                AddReward(-10.0f);
            }
            if (debugMode) Debug.Log(GetCumulativeReward());
            EndEpisode();
        }

    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Target"))
        {
            AddReward(10.0f);
            if (debugMode) Debug.Log(GetCumulativeReward());
            EndEpisode(); 
        }
    }

    // This maps your keyboard to the Action Buffers
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;
        float forwardInput = Input.GetAxis("Vertical"); 
        float turnInput = Input.GetAxis("Horizontal");

        continuousActions[0] = forwardInput; // Throttle
        continuousActions[1] = turnInput;    // Steering
    }

}