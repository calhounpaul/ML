import os

#cuda visible devices= 0,1
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from pydantic import BaseModel
import json
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, logging
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
torch.cuda.empty_cache()
import outlines


MODEL_DIR_PATH="/ephemeral_cache/mistral-inst-v03"


function_json_string = '''{
    "type": "function",
    "function": {
        "name": "store_text_outline",
        "description": "Store the outline of a text in a hierarchical manner",
        "parameters": {
            "type": "object",
            "properties": {
                "main_title": {
                    "type": "string",
                    "description": "A brief title of the work",
                    "minLength": 10
                },
                "main_introduction": {
                    "type": "string",
                    "description": "An introduction to the work",
                    "minLength": 10
                },
                "sections": {
                    "type": "array",
                    "minItems": 2,
                    "items": {
                        "type": "object",
                        "properties": {
                            "section_title": {
                                "type": "string",
                                "description": "A brief title of the section in the work",
                                "minLength": 3
                            },
                            "section_introduction": {
                                "type": "string",
                                "description": "Introduction to the section in the work",
                                "minLength": 10
                            },
                            "subsections": {
                                "type": "array",
                                "minItems": 2,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "subsection_title": {
                                            "type": "string",
                                            "description": "The title of the subsection",
                                            "minLength": 3
                                        },
                                        "subsection_introduction": {
                                            "type": "string",
                                            "description": "Introduction to the subsection",
                                            "minLength": 10
                                        },
                                        "subsection_points": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "description": "Ordered bullet points for the subsection. (do not include the bullet point symbol)",
                                                "minLength": 3
                                            },
                                            "minItems": 1
                                        },
                                        "subsection_conclusion": {
                                            "type": "string",
                                            "description": "A summary conclusion of the subsection",
                                            "minLength": 10
                                        }
                                    },
                                    "required": ["subsection_title", "subsection_introduction", "subsection_points", "subsection_conclusion"]
                                }
                            },
                            "section_conclusion": {
                                "type": "string",
                                "description": "A summary conclusion of the section",
                                "minLength": 10
                            }
                        },
                        "required": ["section_title", "section_introduction", "subsections", "section_conclusion"]
                    }
                },
                "main_conclusion": {
                    "type": "string",
                    "description": "A summary conclusion of the work",
                    "minLength": 10
                }
            },
            "required": ["main_title", "main_introduction", "sections", "main_conclusion"]
        }
    }
}'''

instruction = """Create a hierarchical outline of the following text, including introductions and conclusions for each section:

THE SENTIMENTS OF A CHURCH OF ENGLAND MAN, WITH RESPECT TO RELIGION AND GOVERNMENT.

Whosoever hath examined the conduct and proceedings of both parties for some years past, whether in or out of power, cannot well conceive it possible to go far towards the extremes of either, without offering some violence to his integrity or understanding. A wise and a good man may indeed be sometimes induced to comply with a number whose opinion he generally approves, though it be perhaps against his own. But this liberty should be made use of upon very few occasions, and those of small importance, and then only with a view of bringing over his own side another time to something of greater and more public moment. But to sacrifice the innocency of a friend, the good of our country, or our own conscience to the humour, or passion, or interest of a party, plainly shews that either our heads or our hearts are not as they should be: Yet this very practice is the fundamental law of each faction among us, as may be obvious to any who will impartially, and without engagement, be at the pains to examine their actions, which however is not so easy a task: For it seems a principle in human nature, to incline one way more than another, even in matters where we are wholly unconcerned. And it is a common observation, that in reading a history of facts done a thousand years ago, or standing by at play among those who are perfect strangers to us, we are apt to find our hopes and wishes engaged on a sudden in favour of one side more than another. No wonder then, we are all so ready to interest ourselves in the course of public affairs, where the most inconsiderable have some real share, and by the wonderful importance which every man is of to himself, a very great imaginary one.

And indeed, when the two parties that divide the whole commonwealth, come once to a rupture, without any hopes left of forming a third with better principles, to balance the others; it seems every man’s duty to choose a side, though he cannot entirely approve of either; and all pretences to neutrality are justly exploded by both, being too stale and obvious, only intending the safety and ease of a few individuals while the public is embroiled. This was the opinion and practice of the latter Cato, whom I esteem to have been the wisest and best of all the Romans. But before things proceed to open violence, the truest service a private man may hope to do his country, is, by unbiassing his mind as much as possible, and then endeavouring to moderate between the rival powers; which must needs be owned a fair proceeding with the world, because it is of all others the least consistent with the common design, of making a fortune by the merit of an opinion.

1: Faulkner and Scott have “one of the two sides.” [T. S.]]

I have gone as far as I am able in qualifying myself to be such a moderator: I believe I am no bigot in religion, and I am sure I am none in government. I converse in full freedom with many considerable men of both parties, and if not in equal number, it is purely accidental and personal, as happening to be near the court, and to have made acquaintance there, more under one ministry than another. Then, I am not under the necessity of declaring myself by the prospect of an employment. And lastly, if all this be not sufficient, I industriously conceal my name, which wholly exempts me from any hopes and fears in delivering my opinion.

In consequence of this free use of my reason, I cannot possibly think so well or so ill of either party, as they would endeavour to persuade the world of each other, and of themselves. For instance; I do not charge it upon the body of the Whigs or the Tories, that their several principles lead them to introduce Presbytery, and the religion of the Church of Rome, or a commonwealth and arbitrary power. For, why should any party be accused of a principle which they solemnly disown and protest against? But, to this they have a mutual answer ready; they both assure us, that their adversaries are not to be believed, that they disown their principles out of fear, which are manifest enough when we examine their practices. To prove this, they will produce instances, on one side, either of avowed Presbyterians, or persons of libertine and atheistical tenets, and on the other, of professed Papists, or such as are openly in the interest of the abdicated family. Now, it is very natural for all subordinate sects and denominations in a state, to side with some general party, and to choose that which they find to agree with themselves in some general principle. Thus at the restoration, the Presbyterians, Anabaptists, Independents, and other sects, did all with very good reason unite and solder up their several schemes to join against the Church, who without regard to their distinctions, treated them all as equal adversaries. Thus, our present dissenters do very naturally close in with the Whigs, who profess moderation, declare they abhor all thoughts of persecution, and think it hard that those who differ only in a few ceremonies and speculations, should be denied the privilege and profit of serving their country in the highest employments of state. Thus, the atheists, libertines, despisers of religion and revelation in general, that is to say, all those who usually pass under the name of freethinkers, do properly join with the same body; because they likewise preach up moderation, and are not so overnice to distinguish between an unlimited liberty of conscience, and an unlimited freedom of opinion. Then on the other side, the professed firmness of the Tories for Episcopacy as an apostolical institution: Their aversion to those sects who lie under the reproach of having once destroyed their constitution, and who they imagine, by too indiscreet a zeal for reformation have defaced the primitive model of the Church: Next, their veneration for monarchical government in the common course of succession, and their hatred to republican schemes: These, I say, are principles which not only the nonjuring zealots profess, but even Papists themselves fall readily in with. And every extreme here mentioned flings a general scandal upon the whole body it pretends to adhere to.

But surely no man whatsoever ought in justice or good manners to be charged with principles he actually disowns, unless his practices do openly and without the least room for doubt contradict his profession: Not upon small surmises, or because he has the misfortune to have ill men sometimes agree with him in a few general sentiments. However, though the extremes of Whig and Tory seem with little justice to have drawn religion into their controversies, wherein they have small concern; yet they both have borrowed one leading principle from the abuse of it; which is, to have built their several systems of political faith, not upon enquiries after truth, but upon opposition to each other, upon injurious appellations, charging their adversaries with horrid opinions, and then reproaching them for the want of charity; et neuter falso.

In order to remove these prejudices, I have thought nothing could be more effectual than to describe the sentiments of a Church of England man with respect to religion and government. This I shall endeavour to do in such a manner as may not be liable to least objection from either party, and which I am confident would be assented to by great numbers in both, if they were not misled to those mutual misrepresentations, by such motives as they would be ashamed to own.

I shall begin with religion.

And here, though it makes an odd sound, yet it is necessary to say, that whoever professes himself a member of the Church of England, ought to believe a God and his providence, together with revealed religion, and the divinity of Christ. For beside those many thousands, who (to speak in the phrase of divines) do practically deny all this by the immorality of their lives; there is no small number, who in their conversation and writings directly or by consequence endeavour to overthrow it; yet all these place themselves in the list of the National Church, though at the same time (as it is highly reasonable) they are great sticklers for liberty of conscience.

To enter upon particulars: A Church of England man hath a true veneration for the scheme established among us of ecclesiastic government; and though he will not determine whether Episcopacy be of divine right, he is sure it is most agreeable to primitive institution, fittest of all others for preserving order and purity, and under its present regulations best calculated for our civil state: He should therefore think the abolishment of that order among us would prove a mighty scandal and corruption to our faith, and manifestly dangerous to our monarchy; nay, he would defend it by arms against all the powers on earth, except our own legislature; in which case he would submit as to a general calamity, a dearth, or a pestilence.

As to rites and ceremonies, and forms of prayer; he allows there might be some useful alterations, and more, which in the prospect of uniting Christians might be very supportable, as things declared in their own nature indifferent; to which he therefore would readily comply, if the clergy, or, (though this be not so fair a method) if the legislature should direct: Yet at the same time he cannot altogether blame the former for their unwillingness to consent to any alteration; which beside the trouble, and perhaps disgrace, would certainly never produce the good effects intended by it. The only condition that could make it prudent and just for the clergy to comply in altering the ceremonial or any other indifferent part, would be, a firm resolution in the legislature to interpose by some strict and effectual laws to prevent the rising and spreading of new sects how plausible soever, for the future; else there must never be an end: And it would be to act like a man who should pull down and change the ornaments of his house, in compliance to every one who was disposed to find fault as he passed by, which besides the perpetual trouble and expense, would very much damage, and perhaps in time destroy the building. Sects in a state seem only tolerated with any reason because they are already spread; and because it would not be agreeable with so mild a government, or so pure a religion as ours, to use violent methods against great numbers of mistaken people, while they do not manifestly endanger the constitution of either. But the greatest advocates for general liberty of conscience, will allow that they ought to be checked in their beginnings, if they will allow them to be an evil at all, or which is the same thing, if they will only grant, it were better for the peace of the state, that there should be none. But while the clergy consider the natural temper of mankind in general, or of our own country in particular, what assurances can they have, that any compliances they shall make, will remove the evil of dissension, while the liberty still continues of professing whatever new opinion we please? Or how can it be imagined that the body of dissenting teachers, who must be all undone by such a revolution, will not cast about for some new objections to withhold their flocks, and draw in fresh proselytes by some further innovations or refinements?

Upon these reasons he is for tolerating such different forms in religious worship as are already admitted, but by no means for leaving it in the power of those who are tolerated, to advance their own models upon the ruin of what is already established, which it is natural for all sects to desire, and which they cannot justify by any consistent principles if they do not endeavour; and yet, which they cannot succeed in without the utmost danger to the public peace.

To prevent these inconveniences, he thinks it highly just, that all rewards of trust, profit, or dignity, which the state leaves in the disposal of the administration, should be given only to those whose principles direct them to preserve the constitution in all its parts. In the late affair of Occasional Conformity, the general argument of those who were against it, was not, to deny it an evil in itself, but that the remedy proposed was violent, untimely, and improper, which is the Bishop of Salisbury’s opinion in the speech he made and published against the bill: But, however just their fears or complaints might have been upon that score, he thinks it a little too gross and precipitate to employ their writers already in arguments for repealing the sacramental test, upon no wiser a maxim, than that no man should on the account of conscience be deprived the liberty of serving his country; a topic which may be equally applied to admit Papists, Atheists, Mahometans, Heathens, and Jews. If the Church wants members of its own to employ in the service of the public; or be so unhappily contrived as to exclude from its communion such persons who are likeliest to have great abilities, it is time it should be altered and reduced into some more perfect, or at least more popular form: But in the meanwhile, it is not altogether improbable, that when those who dislike the constitution, are so very zealous in their offers for the service of their country, they are not wholly unmindful of their party or of themselves.

The Dutch whose practice is so often quoted to prove and celebrate the great advantages of a general liberty of conscience, have yet a national religion professed by all who bear office among them: But why should they be a precedent for us either in religion or government? Our country differs from theirs, as well in situation, soil, and productions of nature, as in the genius and complexion of inhabitants. They are a commonwealth founded on a sudden by a desperate attempt in a desperate condition, not formed or digested into a regular system by mature thought and reason, but huddled up under the pressure of sudden exigencies; calculated for no long duration, and hitherto subsisting by accident in the midst of contending powers, who cannot yet agree about sharing it among them. These difficulties do indeed preserve them from any great corruptions, which their crazy constitution would extremely subject them to in a long peace. That confluence of people in a persecuting age, to a place of refuge nearest at hand, put them upon the necessity of trade, to which they wisely gave all ease and encouragement: And if we could think fit to imitate them in this last particular, there would need no more to invite foreigners among us; who seem to think no further than how to secure their property and conscience, without projecting any share in that government which gives them protection, or calling it persecution if it be denied them. But I speak it for the honour of our administration, that although our sects are not so numerous as those in Holland, which I presume is not our fault, and I hope is not our misfortune, we much excel them and all Christendom besides in our indulgence to tender consciences. One single compliance with the national form of receiving the sacrament, is all we require to qualify any sectary among us for the greatest employments in the state, after which he is at liberty to rejoin his own assemblies for the rest of his life. Besides, I will suppose any of the numerous sects in Holland, to have so far prevailed as to have raised a civil war, destroyed their government and religion, and put their administrators to death; after which I will suppose the people to have recovered all again, and to have settled on their old foundation. Then I would put a query, whether that sect which was the unhappy instrument of all this confusion, could reasonably expect to be entrusted for the future with the greatest employments, or indeed to be hardly tolerated among them?

2: When this was written there was no law against Occasional

Conformity. [Faulkner, 1735.]]

To go on with the sentiments of a Church of England man: He does not see how that mighty passion for the Church which some men pretend, can well consist with those indignities and that contempt they bestow on the persons of the clergy. Tis a strange mark whereby to distinguish High Churchmen, that they are such who imagine the clergy can never be too low. He thinks the maxim these gentlemen are so fond of, that they are for an humble clergy, is a very good one; and so is he, and for an humble laity too, since humility is a virtue that perhaps equally benefits and adorns every station of life.

3: “I observed very well with what insolence and haughtiness some lords of the High-Church party treated, not only their own chaplains, but all other clergy whatsoever, and thought this was sufficiently recompensed by their professions of zeal to the church.”]

But then, if the scribblers on the other side freely speak the sentiments of their party, a divine of the Church of England cannot look for much better quarter thence. You shall observe nothing more frequent in their weekly papers than a way of affecting to confound the terms of Clergy and High Church, of applying both indifferently, and then loading the latter with all the calumny they can invent. They will tell you they honour a clergyman; but talk, at the same time, as if there were not three in the kingdom, who could fall in with their definition. After the like manner they insult the universities, as poisoned fountains, and corrupters of youth.

4: “I had likewise observed how the Whig lords took a direct contrary measure, treated the persons of particular clergymen with great courtesy, but shewed much ill-will and contempt for the order in general.”]

Now, it seems clear to me, that the Whigs might easily have procured and maintained a majority among the clergy, and perhaps in the universities, if they had not too much encouraged or connived at this intemperance of speech and virulence of pen, in the worst and most prostitute of their party; among whom there has been for some years past such a perpetual clamour against the ambition, the implacable temper, and the covetousness of the priesthood: Such a cant of High Church, and persecution, and being priestridden; so many reproaches about narrow principles, or terms of communion: Then such scandalous reflections on the universities, for infecting the youth of the nation with arbitrary and Jacobite principles, that it was natural for those, who had the care of religion and education, to apprehend some general design of altering the constitution of both. And all this was the more extraordinary, because it could not easily be forgot, that whatever opposition was made to the usurpations of King James, proceeded altogether from the Church of England, and chiefly from the clergy, and one of the universities. For, if it were of any use to recall matters of fact, what is more notorious than that prince’s applying himself first to the Church of England? And upon their refusal to fall in with his measures, making the like advances to the dissenters of all kinds, who readily and almost universally complied with him, affecting in their numerous addresses and pamphlets, the style of Our Brethren the Roman Catholics, whose interests they put on the same foot with their own: And some of Cromwell’s officers took posts in the army raised against the Prince of Orange. These proceedings of theirs they can only extenuate by urging the provocations they had met from the Church in King Charles’s reign, which though perhaps excusable upon the score of human infirmity, are not by any means a plea of merit equal to the constancy and sufferings of the bishops and clergy, or of the head and fellows of Magdalen College, that furnished the Prince of Orange’s declaration with such powerful arguments to justify and promote the Revolution.

5: De Foe’s “History of Addresses” contains some humbling instances of the applause with which the sectaries hailed their old enemy, James II., when they saw him engaged in hostility with the established Church. [T. S.]]



Therefore a Church of England man abhors the humour of the age in delighting to fling scandals upon the clergy in general; which besides the disgrace to the Reformation, and to religion itself, casts an ignominy upon the kingdom that it does not deserve. We have no better materials to compound the priesthood of, than the mass of mankind, which corrupted as it is, those who receive orders must have some vices to leave behind them when they enter into the Church, and if a few do still adhere, it is no wonder, but rather a great one that they are no worse. Therefore he cannot think ambition, or love of power more justly laid to their charge than to other men, because, that would be to make religion itself, or at least the best constitution of Church-government, answerable for the errors and depravity of human nature.

Within these last two hundred years all sorts of temporal power have been wrested from the clergy, and much of their ecclesiastic, the reason or justice of which proceeding I shall not examine; but, that the remedies were a little too violent with respect to their possessions, the legislature hath lately confessed by the remission of their First Fruits. Neither do the common libellers deny this, who in their invectives only tax the Church with an insatiable desire of power and wealth (equally common to all bodies of men as well as individuals) but thank God, that the laws have deprived them of both. However, it is worth observing the justice of parties: The sects among us are apt to complain, and think it hard usage to be reproached now after fifty years for overturning the state, for the murder of a king, and the indignity of a usurpation; yet these very men and their partisans, are continually reproaching the clergy, and laying to their charge the pride, the avarice, the luxury, the ignorance, and superstition, of Popish times for a thousand years past.

6: The first fruits were the first year’s income of ecclesiastical benefices. In the middle ages they were taken by the Pope as a right; but were handed over to the English crown in 1534. Anne in 1703 gave them back to the Church by letters patent, an act confirmed by Parliament in 1704. The “Bounty” of Queen Anne, however, did not extend to Ireland; and one of Swift’s missions in London was to obtain this remission of the first fruits for the Irish clergy also. [T. S.]]

He thinks it a scandal to government that such an unlimited liberty should be allowed of publishing books against those doctrines in religion, wherein all Christians have agreed, much more to connive at such tracts as reject all revelation, and by their consequences often deny the very being of a God. Surely ’tis not a sufficient atonement for the writers, that they profess much loyalty to the present government, and sprinkle up and down some arguments in favour of the dissenters; that they dispute as strenuously as they can for liberty of conscience, and inveigh largely against all ecclesiastics, under the name of High Church; and, in short, under the shelter of some popular principles in politics and religion, undermine the foundations of all piety and virtue.

As he doth not reckon every schism of that damnable nature which some would represent, so he is very far from closing with the new opinion of those who would make it no crime at all, and argue at a wild rate, that God Almighty is delighted with the variety of faith and worship, as He is with the varieties of nature. To such absurdities are men carried by the affectation of freethinking, and removing the prejudices of education, under which head they have for some time begun to list morality and religion. It is certain that before the rebellion in 1642, though the number of Puritans (as they were then called) was as great as it is with us, and though they affected to follow pastors of that denomination, yet those pastors had episcopal ordination, possessed preferments in the Church, and were sometimes promoted to bishoprics themselves. But, a breach in the general form of worship was in those days reckoned so dangerous and sinful in itself, and so offensive to Roman Catholics at home and abroad, and that it was too unpopular to be attempted; neither, I believe, was the expedient then found out of maintaining separate pastors out of private purses.

7: In the reign of Elizabeth, and even in that of James, the

Puritans were not, properly speaking, Dissenters; but, on the contrary,

formed a sort of Low Church party in the national establishment.

Archbishop Abbot himself has been considered as a Puritan. [T. S.]]

When a schism is once spread in a nation, there grows at length a dispute which are the schismatics. Without entering on the arguments, used by both sides among us, to fix the guilt on each other; ’tis certain, that, in the sense of the law, the schism lies on that side which opposes itself to the religion of the state. I leave it among the divines to dilate upon the danger of schism, as a spiritual evil, but I would consider it only as a temporal one. And I think it clear that any great separation from the established worship, though to a new one that is more pure and perfect, may be an occasion of endangering the public peace, because it will compose a body always in reserve, prepared to follow any discontented heads upon the plausible pretext of advancing true religion, and opposing error, superstition, or idolatry. For this reason Plato lays it down as a maxim, that, men ought to worship the gods according to the laws of the country, and he introduces Socrates in his last discourse utterly disowning the crime laid to his charge, of teaching new divinities or methods of worship. Thus the poor Huguenots of France were engaged in a civil war, by the specious pretences of some, who under the guise of religion sacrificed so many thousand lives to their own ambition and revenge. Thus was the whole body of Puritans in England drawn to be instruments, or abettors of all manner of villainy, by the artifices of a few men whose designs from the first were levelled to destroy the constitution both of religion and government. And thus, even in Holland itself, where it is pretended that the variety of sects live so amicably together, and in such perfect obedience to the magistrate, it is notorious how a turbulent party joining with the Arminians, did in the memory of our fathers attempt to destroy the liberty of that republic. So that upon the whole, where sects are tolerated in a state, ’tis fit they should enjoy a full liberty of conscience, and every other privilege of freeborn subjects to which no power is annexed. And to preserve their obedience upon all emergencies, a government cannot give them too much ease, nor trust them with too little power.

8: Lord Clarendon’s History; but see also Gardiner’s “History of England.” [T. S.]]



The clergy are usually charged with a persecuting spirit, which they are said to discover by an implacable hatred to all dissenters; and this appears to be more unreasonable, because they suffer less in their interests by a toleration than any of the conforming laity: For while the Church remains in its present form, no dissenter can possibly have any share in its dignities, revenues, or power; whereas, by once receiving the sacrament, he is rendered capable of the highest employments in the state. And it is very possible, that a narrow education, together with a mixture of human infirmity, may help to beget among some of the clergy in possession such an aversion and contempt for all innovators, as physicians are apt to have for empirics, or lawyers for pettifoggers, or merchants for pedlars: But since the number of sectaries doth not concern the clergy either in point of interest or conscience, (it being an evil not in their power to remedy) ’tis more fair and reasonable to suppose their dislike proceeds from the dangers they apprehend to the peace of the commonwealth, in the ruin whereof they must expect to be the first and greatest sufferers.

To conclude this section, it must be observed, there is a very good word, which hath of late suffered much by both parties, and that is, MODERATION, which the one side very justly disowns, and the other as unjustly pretends to. Beside what passeth every day in conversation; any man who reads the papers published by Mr. Lesley and others of his stamp, must needs conclude, that if this author could make the nation see his adversaries under the colours he paints them in, we have nothing else to do, but rise as one man and destroy such wretches from the face of the earth. On the other side, how shall we excuse the advocates for moderation? among whom, I could appeal to a hundred papers of universal approbation by the cause they were writ for, which lay such principles to the whole body of the Tories, as, if they were true, and believed; our next business should in prudence be, to erect gibbets in every parish, and hang them out of the way. But I suppose it is presumed, the common people understand raillery, or at least, rhetoric, and will not take hyperboles in too literal a sense; which however in some junctures might prove a desperate experiment.

9: This was Charles Leslie, the second son of the Bishop of Clogher (1650-1722). He was educated for the bar, but forsook that, and entered into holy orders. In his zeal for the established Church he persecuted the Catholics; but this did not interfere with his adhesion to Jacobite political principles. He settled in London, and wrote a weekly paper called “The Rehearsal, or a Review of the Times,” in which he attacked Locke and Hoadly. He did all he could for the cause of the exiled James, but he gave up the work when he found it hopeless, and died in Ireland. He wrote many virulent theological works, as well as a host of political tracts. [T. S.]]

And this is moderation in the modern sense of the word, to which, speaking impartially, the bigots of both parties are equally entitled."""

function_json = json.loads(function_json_string)

json_string = json.dumps(function_json)

tools_string="[AVAILABLE_TOOLS] ["+json_string+"][/AVAILABLE_TOOLS]"
instruction_string="[INST] "+instruction+" [/INST]"
prompt = tools_string+instruction_string + "[TOOL_CALLS]"

function_params=function_json["function"]["parameters"].copy()
function_params["title"] = function_json["function"]["name"]+"_input_params"
function_params_string = json.dumps(function_params)

config = AutoConfig.from_pretrained(MODEL_DIR_PATH)

print("init empty weights")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

print("infer auto device map")

max_memory = {0:"4.5GiB", 1:"4.5GiB","cpu":"0GiB"}


print("max_memory", max_memory)

device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["MistralDecoderLayer"],
        dtype=torch.int8
    )

print(device_map)
input()

print("load checkpoint and dispatch")
model = outlines.models.transformers(
        MODEL_DIR_PATH,
        #device="cuda",   #why... https://github.com/outlines-dev/outlines/blob/a987159860a6dd3a83d2f2376f36ab28ef45decd/outlines/models/transformers.py#L229
        device=None,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "device_map": device_map,
            "load_in_8bit": True,
            "config":config,
        },
    )
generator = outlines.generate.json(model, function_params_string)
output = generator(prompt)

print(json.dumps(output, indent=2))

with open('example_output.json', 'w') as f:
    json.dump(output, f, indent=2)